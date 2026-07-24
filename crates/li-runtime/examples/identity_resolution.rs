//! Random example code made by AI.
//! It works and creates a HTML code.

use std::boxed::Box;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::vec::Vec;

use li_core::belief::BeliefState;
use li_core::events::RuntimeEvent;
use li_core::ids::{IdentityId, ObservationId, VertexId};
use li_core::observation::{Evidence, Modality, Observation, Timestamp};
use li_core::probability::{Confidence, Probability};
use li_core::relation::Relation;
use li_factors::compiler::FactorCompiler;
use li_factors::factor::{Factor, FactorScope};
use li_model::ontology::IdentityNode;
use li_model::operations::GraphOperation;
use li_runtime::engine::{EngineConfig, RuntimeEngine};
use li_runtime::executor::ExecutionSink;
use li_workspace::checkpoint::WorkspaceSnapshot;
use li_workspace::workspace::ActiveWorkspace;

/// Payload emitted by a fixed camera sensor snapshot
#[derive(Debug, Clone, PartialEq)]
pub struct CameraSnapshotPayload {
    pub camera_id: u32,
    pub camera_name: &'static str,
    pub cam_x: f64,
    pub cam_y: f64,
    pub person_x: f64,
    pub person_y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackSummary {
    pub last_x: f64,
    pub last_y: f64,
    pub last_timestamp: Timestamp,
}

pub struct SpatioTemporalFactor {
    pub candidate_id: IdentityId,
    pub obs_x: f64,
    pub obs_y: f64,
    pub obs_time: i64,
    pub belief_x: f64,
    pub belief_y: f64,
    pub belief_time: i64,
}

impl FactorScope for SpatioTemporalFactor {
    fn scope(&self) -> &[IdentityId] {
        core::slice::from_ref(&self.candidate_id)
    }
}

impl Factor for SpatioTemporalFactor {
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability {
        if assignment.is_empty() || assignment[0] != self.candidate_id {
            return Probability::new(0.01);
        }

        let dx = self.obs_x - self.belief_x;
        let dy = self.obs_y - self.belief_y;
        let dist = (dx * dx + dy * dy).sqrt();

        let dt_sec = ((self.obs_time - self.belief_time).abs().max(1)) as f64 /
            1_000_000.0;
        let speed = if dt_sec > 0.0 { dist / dt_sec } else { 0.0 };

        if speed > 45.0 {
            Probability::new(0.001)
        } else {
            let likelihood = (-dist / 50.0).exp();
            Probability::new(likelihood.clamp(0.01, 0.99))
        }
    }
}

pub struct SpatioTemporalFactorCompiler;

impl FactorCompiler<CameraSnapshotPayload, TrackSummary>
    for SpatioTemporalFactorCompiler
{
    fn compile_factors(
        &self,
        evidence: &Evidence<CameraSnapshotPayload>,
        active_beliefs: &[BeliefState<TrackSummary>],
    ) -> Vec<Box<dyn Factor>> {
        let mut factors: Vec<Box<dyn Factor>> = Vec::new();

        for &cand_id in &evidence.candidates {
            if let Some(belief) =
                active_beliefs.iter().find(|b| b.identity == cand_id)
            {
                factors.push(Box::new(SpatioTemporalFactor {
                    candidate_id: cand_id,
                    obs_x: evidence.observation.payload.person_x,
                    obs_y: evidence.observation.payload.person_y,
                    obs_time: evidence.observation.timestamp.0,
                    belief_x: belief.summary.last_x,
                    belief_y: belief.summary.last_y,
                    belief_time: belief.summary.last_timestamp.0,
                }));
            }
        }

        factors
    }
}

#[derive(Default)]
pub struct InMemoryWorkspace {
    beliefs: Vec<BeliefState<TrackSummary>>,
}

impl ActiveWorkspace for InMemoryWorkspace {
    type Summary = TrackSummary;

    fn insert(&mut self, belief: BeliefState<Self::Summary>) {
        if let Some(existing) = self
            .beliefs
            .iter_mut()
            .find(|b| b.identity == belief.identity)
        {
            *existing = belief;
        } else {
            self.beliefs.push(belief);
        }
    }

    fn get(&self, id: IdentityId) -> Option<&BeliefState<Self::Summary>> {
        self.beliefs.iter().find(|b| b.identity == id)
    }

    fn get_mut(
        &mut self,
        id: IdentityId,
    ) -> Option<&mut BeliefState<Self::Summary>> {
        self.beliefs.iter_mut().find(|b| b.identity == id)
    }

    fn active_beliefs(&self) -> Vec<&BeliefState<Self::Summary>> {
        self.beliefs.iter().collect()
    }

    fn evict_expired(
        &mut self,
        _current_time: Timestamp,
        _ttl: i64,
    ) -> Vec<BeliefState<Self::Summary>> {
        Vec::new()
    }

    fn create_snapshot(
        &self,
        current_time: Timestamp,
    ) -> WorkspaceSnapshot<Self::Summary> {
        WorkspaceSnapshot {
            timestamp: current_time,
            active_states: self.beliefs.clone(),
        }
    }
}

#[derive(Default, Debug)]
pub struct GraphStoreSink {
    pub identities: Vec<IdentityNode>,
    pub observations: Vec<Observation<CameraSnapshotPayload>>,
    pub relations: Vec<(VertexId, Relation, VertexId, Timestamp)>,
}

impl ExecutionSink<CameraSnapshotPayload, (), TrackSummary>
    for GraphStoreSink
{
    type Error = ();

    fn execute_batch(
        &mut self,
        operations: &[GraphOperation<
            CameraSnapshotPayload,
            (),
            TrackSummary,
        >],
    ) -> Result<(), Self::Error> {
        for op in operations {
            match op {
                GraphOperation::CommitIdentity(node) => {
                    self.identities.push(node.clone());
                },
                GraphOperation::CommitObservation(obs) => {
                    self.observations.push(obs.clone());
                },
                GraphOperation::CommitRelation {
                    source,
                    relation,
                    target,
                    created_at,
                } => {
                    self.relations.push((
                        *source,
                        *relation,
                        *target,
                        *created_at,
                    ));
                },
                _ => {},
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FixedCamera {
    pub id: u32,
    pub name: &'static str,
    pub x: f64,
    pub y: f64,
}

const BASE_TIME: i64 = 1_700_000_000_000_000;

fn main() {
    println!(
        "=== LI-ESKG: Multi-Camera Surveillance & Pedestrian Resolution ==="
    );

    let cameras = vec![
        FixedCamera {
            id: 1,
            name: "Cam-01 Main Gate",
            x: 0.0,
            y: 0.0,
        },
        FixedCamera {
            id: 2,
            name: "Cam-02 North Plaza",
            x: 250.0,
            y: 180.0,
        },
        FixedCamera {
            id: 3,
            name: "Cam-03 East Boulevard",
            x: 500.0,
            y: 380.0,
        },
        FixedCamera {
            id: 4,
            name: "Cam-04 West Avenue",
            x: -300.0,
            y: 220.0,
        },
        FixedCamera {
            id: 5,
            name: "Cam-05 South Station",
            x: -150.0,
            y: -250.0,
        },
        FixedCamera {
            id: 6,
            name: "Cam-06 Park Crossing",
            x: 320.0,
            y: -380.0,
        },
        FixedCamera {
            id: 7,
            name: "Cam-07 Highway Overpass",
            x: 700.0,
            y: -80.0,
        },
    ];

    let config = EngineConfig {
        decision_threshold: 0.40,
        direct_assignment_threshold: 0.95,
    };

    let workspace = InMemoryWorkspace::default();
    let compiler = SpatioTemporalFactorCompiler;
    let sink = GraphStoreSink::default();

    let mut engine =
        RuntimeEngine::new(config, 100, workspace, compiler, sink);
    let mut raw_observations = Vec::new();
    let mut obs_counter = 1u64;

    let mut add_pedestrian_path =
        |start_x: f64,
         start_y: f64,
         vx: f64,
         vy: f64,
         dt_sec: f64,
         steps: usize| {
            for i in 0..steps {
                let t = BASE_TIME + (i as f64 * dt_sec * 1_000_000.0) as i64;
                let px = start_x + vx * (i as f64 * dt_sec);
                let py = start_y + vy * (i as f64 * dt_sec);

                if let Some(cam) = cameras.iter().find(|c| {
                    let dx = c.x - px;
                    let dy = c.y - py;
                    (dx * dx + dy * dy).sqrt() < 130.0
                }) {
                    let noise_x = (i as f64 * 1.7).sin() * 0.8;
                    let noise_y = (i as f64 * 2.1).cos() * 0.8;

                    raw_observations.push(Observation {
                        id: ObservationId(obs_counter),
                        modality: Modality(1),
                        timestamp: Timestamp(t),
                        confidence: Confidence(0.92 + (i % 3) as f64 * 0.02),
                        payload: CameraSnapshotPayload {
                            camera_id: cam.id,
                            camera_name: cam.name,
                            cam_x: cam.x,
                            cam_y: cam.y,
                            person_x: px + noise_x,
                            person_y: py + noise_y,
                        },
                    });
                    obs_counter += 1;
                }
            }
        };

    add_pedestrian_path(-20.0, -15.0, 18.5, 13.5, 4.0, 7); // Pedestrian 1
    add_pedestrian_path(530.0, 400.0, -22.0, -16.0, 4.5, 6); // Pedestrian 2
    add_pedestrian_path(-310.0, 240.0, 12.0, -32.0, 3.5, 7); // Pedestrian 3
    add_pedestrian_path(310.0, -390.0, 28.0, 22.0, 5.0, 6); // Pedestrian 4

    raw_observations.sort_by_key(|obs| obs.timestamp.0);

    println!(
        "Ingesting {} camera snapshots from fixed sensors...",
        raw_observations.len()
    );

    for obs in raw_observations.clone() {
        let candidates: Vec<IdentityId> = engine
            .workspace()
            .active_beliefs()
            .iter()
            .filter(|b| {
                let dx = obs.payload.person_x - b.summary.last_x;
                let dy = obs.payload.person_y - b.summary.last_y;
                (dx * dx + dy * dy).sqrt() < 220.0
            })
            .map(|b| b.identity)
            .collect();

        let evidence = Evidence {
            observation: obs.clone(),
            candidates,
        };

        engine
            .submit_event(RuntimeEvent::Observation(evidence))
            .unwrap();
        let processed = engine.tick::<()>().expect("Tick failed");
        assert!(processed);

        if let Some(last_rel) = engine.executor().sink().relations.last() {
            let assigned_identity = IdentityId(last_rel.2.0);
            engine.workspace_mut().insert(BeliefState {
                identity: assigned_identity,
                summary: TrackSummary {
                    last_x: obs.payload.person_x,
                    last_y: obs.payload.person_y,
                    last_timestamp: obs.timestamp,
                },
                posterior: Probability::new(0.95),
                last_update: obs.timestamp,
            });
        }
    }

    let sink = engine.executor().sink();

    println!("=== Execution Summary ===");
    println!("Total Resolved Individuals: {}", sink.identities.len());
    println!(
        "Total Captured Camera Snapshots: {}",
        sink.observations.len()
    );
    println!("Total Graph Relations: {}", sink.relations.len());

    generate_html_graph_visualization(&cameras, sink, "identity_graph.html");
}

fn generate_html_graph_visualization(
    cameras: &[FixedCamera],
    sink: &GraphStoreSink,
    output_path: &str,
) {
    let mut html = String::new();

    let mut identity_obs_map: HashMap<
        u64,
        Vec<&Observation<CameraSnapshotPayload>>,
    > = HashMap::new();
    for (src, rel, tgt, _) in &sink.relations {
        if *rel == Relation::Supports {
            if let Some(obs) =
                sink.observations.iter().find(|o| o.id.0 == src.0)
            {
                identity_obs_map.entry(tgt.0).or_default().push(obs);
            }
        }
    }
    for list in identity_obs_map.values_mut() {
        list.sort_by_key(|o| o.timestamp.0);
    }

    html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LI-ESKG Multi-Camera Surveillance Visualizer</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --border: #1f2937;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #f59e0b;
      --cctv-green: #22c55e;
    }
    body { font-family: system-ui, -apple-system, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    h1 { margin: 0; font-size: 22px; color: var(--primary); display: flex; align-items: center; gap: 10px; }
    .subtitle { color: var(--muted); font-size: 13px; margin-top: 4px; }
    
    .nav-tabs { display: flex; gap: 10px; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: transparent; border: 1px solid transparent; padding: 8px 16px; font-weight: 600; font-size: 13px; color: var(--muted); cursor: pointer; border-radius: 6px; transition: all 0.2s; }
    .tab-btn.active { background: #1e293b; color: var(--primary); border-color: var(--border); }
    .tab-btn:hover:not(.active) { background: #1e293b55; color: var(--text); }

    .control-panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
    .btn { background: var(--primary); color: #0f172a; border: none; padding: 8px 18px; border-radius: 6px; font-weight: 700; cursor: pointer; transition: all 0.2s; }
    .btn:hover { background: var(--primary-hover); color: white; }
    .btn-secondary { background: #1e293b; color: var(--text); border: 1px solid var(--border); }
    .btn-secondary:hover { background: #334155; }
    .slider-container { flex-grow: 1; display: flex; align-items: center; gap: 12px; }
    input[type=range] { flex-grow: 1; accent-color: var(--primary); }
    .time-badge { background: #0f172a; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-family: monospace; font-weight: bold; font-size: 13px; color: var(--primary); }

    .main-grid { display: grid; grid-template-columns: 1fr 360px; gap: 20px; }
    
    .view-content { display: none; background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    .view-content.active { display: block; }
    #network { width: 100%; height: 620px; background: #070a12; border-radius: 8px; }

    .inspector-panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    .inspector-title { font-size: 13px; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .inspector-card { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.5; }
    .inspector-card b { color: var(--text); }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .badge-tag { background: #0284c7; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }

    /* Custom CCTV Tooltip Popup Overlay */
    #cctvModal {
      position: absolute;
      display: none;
      width: 280px;
      background: #090d16;
      border: 1px solid #22c55e;
      border-radius: 8px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.8), 0 0 15px rgba(34,197,94,0.2);
      font-family: 'Courier New', monospace;
      z-index: 9999;
      pointer-events: none;
      overflow: hidden;
    }
    .cctv-header {
      background: #111827;
      padding: 6px 10px;
      font-size: 11px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f2937;
      color: #22c55e;
      font-weight: bold;
    }
    .cctv-rec { color: #ef4444; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .cctv-feed { width: 100%; height: 160px; object-fit: cover; filter: contrast(1.1) grayscale(0.3); display: block; }
    .cctv-footer { padding: 8px; font-size: 11px; color: #9ca3af; background: #090d16; }

    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th { background: #0f172a; padding: 10px; font-weight: 600; border-bottom: 1px solid var(--border); color: var(--primary); }
    td { padding: 10px; border-bottom: 1px solid var(--border); }
    tr:nth-child(even) { background: #0f172a33; }

    .chart-wrapper { height: 450px; position: relative; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>📹 LI-ESKG Multi-Camera Surveillance</h1>
      <div class="subtitle">Spatiotemporal tracking & dynamic trajectory resolution</div>
    </div>
  </header>

  <div class="control-panel">
    <button id="playBtn" class="btn">▶ Start Sequence</button>
    <button id="stepBackBtn" class="btn btn-secondary">⏮ Previous</button>
    <button id="stepFwdBtn" class="btn btn-secondary">⏭ Next Event</button>
    
    <div class="slider-container">
      <span style="font-size: 13px; font-weight: 600; color: var(--muted);">Timeline:</span>
      <input type="range" id="timeSlider" min="0" value="0">
    </div>

    <div class="time-badge" id="timeDisplay">Event 0 / 0</div>

    <select id="speedSelect" style="padding: 6px 10px; border-radius: 6px; background: #0f172a; border: 1px solid var(--border); color: var(--text); font-size: 13px;">
      <option value="2500">Slow (2.5s)</option>
      <option value="1500" selected>Normal (1.5s)</option>
      <option value="700">Fast (0.7s)</option>
    </select>
  </div>

  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('graphTab', this)">🕸️ Spatial Graph & Cameras</button>
    <button class="tab-btn" onclick="switchTab('tableTab', this)">📋 Ingestion Log</button>
    <button class="tab-btn" onclick="switchTab('chartTab', this)">📈 Trajectory Probabilities</button>
  </div>

  <div id="graphTab" class="view-content active">
    <div class="main-grid">
      <div style="position: relative;">
        <div id="network"></div>
        <!-- Dynamic CCTV Surveillance Camera Tooltip -->
        <div id="cctvModal">
          <div class="cctv-header">
            <span id="cctvCamName">CAM-01</span>
            <span class="cctv-rec">● REC</span>
          </div>
          <img id="cctvImage" class="cctv-feed" src="" alt="CCTV Feed" />
          <div class="cctv-footer">
            <div>Subject: <b id="cctvSubject" style="color: #38bdf8;">Individual #1001</b></div>
            <div>Position: <span id="cctvCoords">(0.0, 0.0)</span></div>
            <div>Confidence: <span id="cctvProb" style="color:#22c55e;">95%</span></div>
          </div>
        </div>
      </div>

      <div class="inspector-panel">
        <div class="inspector-title">🔍 LI-ESKG Decision Inspector</div>
        <div id="inspectorContent">
          <div style="color: var(--muted); font-style: italic; text-align: center; margin-top: 40px;">Click ▶ Start Sequence to analyze individual movements.</div>
        </div>
      </div>
    </div>
  </div>

  <div id="tableTab" class="view-content">
    <table>
      <thead>
        <tr>
          <th>Step</th>
          <th>Camera Sensor</th>
          <th>Snapshot ID</th>
          <th>Time Offset</th>
          <th>Coordinates</th>
          <th>Identified Individual</th>
          <th>Speed (m/s)</th>
          <th>Likelihood</th>
        </tr>
      </thead>
      <tbody id="tableBody"></tbody>
    </table>
  </div>

  <div id="chartTab" class="view-content">
    <div class="chart-wrapper">
      <canvas id="probChart"></canvas>
    </div>
  </div>

  <script type="text/javascript">
"#);

    let cctv_images = vec![
        "https://i.imgur.com/YiquyQI.jpeg",
        "https://i.imgur.com/C1W26GY.jpeg",
        "https://i.imgur.com/TQpTwzY.jpeg",
        "https://i.imgur.com/8BgTcwb.jpeg",
    ];
    let mut cams_json = String::new();
    cams_json.push_str("    const fixedCameras = [\n");
    for cam in cameras {
        cams_json.push_str(&format!(
            "      {{ id: {}, name: \"{}\", x: {:.1}, y: {:.1} }},\n",
            cam.id, cam.name, cam.x, cam.y
        ));
    }
    cams_json.push_str("    ];\n\n");
    html.push_str(&cams_json);

    let mut steps_json = String::new();
    steps_json.push_str("    const simulationSteps = [\n");

    let mut step_idx = 0;
    for (src, rel, tgt, _) in &sink.relations {
        if *rel == Relation::Supports {
            let obs =
                sink.observations.iter().find(|o| o.id.0 == src.0).unwrap();
            let obs_list = identity_obs_map.get(&tgt.0).unwrap();
            let idx =
                obs_list.iter().position(|o| o.id.0 == obs.id.0).unwrap();

            let t_sec = (obs.timestamp.0 - BASE_TIME) as f64 / 1_000_000.0;
            let img_url = cctv_images[step_idx % cctv_images.len()];

            let (speed, likelihood, dt_sec, dist) = if idx > 0 {
                let prev_obs = obs_list[idx - 1];
                let dx = obs.payload.person_x - prev_obs.payload.person_x;
                let dy = obs.payload.person_y - prev_obs.payload.person_y;
                let d = (dx * dx + dy * dy).sqrt();
                let dt = ((obs.timestamp.0 - prev_obs.timestamp.0).abs()
                    as f64) /
                    1_000_000.0;
                let s = if dt > 0.0 { d / dt } else { 0.0 };
                let l = (-d / 50.0).exp().clamp(0.01, 0.99);
                (s, l, dt, d)
            } else {
                (0.0, 0.95, 0.0, 0.0)
            };

            steps_json.push_str(&format!(
                "      {{ step: {}, obsId: {}, identityId: {}, camId: {}, camName: \"{}\", camX: {:.1}, camY: {:.1}, personX: {:.1}, personY: {:.1}, tSec: {:.1}, speed: {:.1}, likelihood: {:.3}, dt: {:.2}, dist: {:.2}, imgUrl: \"{}\" }},\n",
                step_idx, obs.id.0, tgt.0, obs.payload.camera_id, obs.payload.camera_name, obs.payload.cam_x, obs.payload.cam_y, obs.payload.person_x, obs.payload.person_y, t_sec, speed, likelihood, dt_sec, dist, img_url
            ));
            step_idx += 1;
        }
    }
    steps_json.push_str("    ];\n\n");
    html.push_str(&steps_json);

    html.push_str(r#"
    let currentStep = -1;
    let isPlaying = false;
    let timer = null;

    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);

    const container = document.getElementById('network');
    const network = new vis.Network(container, { nodes: nodes, edges: edges }, {
      physics: { enabled: false },
      interaction: { hover: true },
      edges: { font: { align: 'horizontal', color: '#9ca3af', strokeWidth: 0, size: 10 }, width: 2 },
      nodes: { font: { color: '#f3f4f6', face: 'system-ui' } }
    });

    const ctx = document.getElementById('probChart').getContext('2d');
    const probChart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: 'Likelihood Evolution per Pedestrian', color: '#f3f4f6' } },
        scales: { 
          y: { min: 0, max: 1, grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } },
          x: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } }
        }
      }
    });

    function initUI() {
      const slider = document.getElementById('timeSlider');
      slider.max = simulationSteps.length - 1;
      slider.addEventListener('input', (e) => {
        setStep(parseInt(e.target.value));
      });

      document.getElementById('playBtn').addEventListener('click', togglePlay);
      document.getElementById('stepFwdBtn').addEventListener('click', () => setStep(currentStep + 1));
      document.getElementById('stepBackBtn').addEventListener('click', () => setStep(currentStep - 1));

      fixedCameras.forEach(cam => {
        nodes.add({
          id: 500000 + cam.id,
          label: `📷 ${cam.name}`,
          shape: 'box',
          x: cam.x * 1.2,
          y: -cam.y * 1.2,
          color: { background: '#1e293b', border: '#38bdf8' },
          font: { color: '#ffffff', weight: 'bold', size: 12 },
          size: 22
        });
      });

      network.on("hoverNode", function (params) {
        const nodeId = params.node;
        const modal = document.getElementById('cctvModal');

        if (nodeId < 100000) {
          const personSteps = simulationSteps.slice(0, currentStep + 1).filter(s => s.identityId === nodeId);
          if (personSteps.length > 0) {
            const last = personSteps[personSteps.length - 1];
            document.getElementById('cctvCamName').innerText = last.camName;
            document.getElementById('cctvSubject').innerText = `Individual #${nodeId}`;
            document.getElementById('cctvCoords').innerText = `(${last.personX}, ${last.personY})`;
            document.getElementById('cctvProb').innerText = `${(last.likelihood * 100).toFixed(1)}%`;
            document.getElementById('cctvImage').src = last.imgUrl;

            const domPos = network.canvasToDOM(network.getPositions([nodeId])[nodeId]);
            modal.style.left = (domPos.x + 20) + 'px';
            modal.style.top = (domPos.y - 80) + 'px';
            modal.style.display = 'block';
          }
        }
      });

      network.on("blurNode", function () {
        document.getElementById('cctvModal').style.display = 'none';
      });

      setStep(0);
      network.fit({ animation: false });
    }

    function setStep(targetStep) {
      if (targetStep < 0) targetStep = 0;
      if (targetStep >= simulationSteps.length) targetStep = simulationSteps.length - 1;

      document.getElementById('timeSlider').value = targetStep;
      document.getElementById('timeDisplay').innerText = `Event ${targetStep + 1} / ${simulationSteps.length}`;

      updateIncrementalGraph(targetStep);
      updateInspector(targetStep);
      updateTable(targetStep);
      updateChart(targetStep);

      currentStep = targetStep;
    }

    function updateIncrementalGraph(targetStep) {
      const activeSteps = simulationSteps.slice(0, targetStep + 1);
      const activeIdentities = new Set(activeSteps.map(s => s.identityId));

      activeIdentities.forEach(id => {
        const personSteps = activeSteps.filter(s => s.identityId === id);
        const lastStep = personSteps[personSteps.length - 1];

        if (!nodes.get(id)) {
          nodes.add({
            id: id,
            label: `🚶 Individual #${id}\n[${personSteps.length} Snapshots]`,
            shape: 'ellipse',
            x: lastStep.personX * 1.2,
            y: -lastStep.personY * 1.2,
            color: { background: '#10b981', border: '#059669' },
            font: { color: '#ffffff', weight: 'bold' },
            size: 26
          });
        } else {
          nodes.update({
            id: id,
            label: `🚶 Individual #${id}\n[${personSteps.length} Snapshots]`,
            x: lastStep.personX * 1.2,
            y: -lastStep.personY * 1.2
          });
        }

        const camNodeId = 500000 + lastStep.camId;
        const camEdgeId = `cam_link_${id}`;
        if (!edges.get(camEdgeId)) {
          edges.add({
            id: camEdgeId,
            from: camNodeId,
            to: id,
            dashes: true,
            color: { color: '#38bdf8' },
            width: 1.5
          });
        } else {
          edges.update({
            id: camEdgeId,
            from: camNodeId,
            to: id
          });
        }
      });
    }

    function updateInspector(step) {
      const s = simulationSteps[step];
      const panel = document.getElementById('inspectorContent');

      panel.innerHTML = `
        <div class="inspector-card">
          <div class="stat-row"><b>📷 Camera Sensor:</b> <span class="badge-tag">${s.camName}</span></div>
          <div class="stat-row"><b>📸 Snapshot ID:</b> <span>Obs #${s.obsId}</span></div>
          <div class="stat-row"><span>Elapsed Time:</span> <b>T + ${s.tSec}s</b></div>
          <div class="stat-row"><span>Coordinates:</span> <b>(${s.personX}, ${s.personY})</b></div>
        </div>

        <div class="inspector-card" style="border-left: 3px solid #38bdf8;">
          <b>📐 Kinematic Evaluation</b>
          <div style="margin-top: 6px;">
            <div class="stat-row"><span>Interval (Δt):</span> <b>${s.dt}s</b></div>
            <div class="stat-row"><span>Distance Covered:</span> <b>${s.dist}m</b></div>
            <div class="stat-row"><span>Estimated Speed:</span> <b style="color:#38bdf8;">${s.speed} m/s</b></div>
          </div>
        </div>

        <div class="inspector-card" style="border-left: 3px solid #22c55e; background: #052e1633;">
          <b>✅ Identity Graph Association</b>
          <div style="margin-top: 6px;">
            <div class="stat-row"><span>Assigned Target:</span> <b style="color:#22c55e;">Individual #${s.identityId}</b></div>
            <div class="stat-row"><span>Likelihood Score:</span> <b style="color:#22c55e;">${(s.likelihood * 100).toFixed(1)}%</b></div>
          </div>
        </div>
      `;
    }

    function updateTable(step) {
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = '';
      const activeSteps = simulationSteps.slice(0, step + 1);

      activeSteps.forEach((s, idx) => {
        const tr = document.createElement('tr');
        if (idx === step) tr.style.background = '#0284c733';
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><b>${s.camName}</b></td>
          <td>📸 Obs #${s.obsId}</td>
          <td>+${s.tSec}s</td>
          <td>(${s.personX}, ${s.personY})</td>
          <td><span style="color:#22c55e; font-weight:bold;">Individual #${s.identityId}</span></td>
          <td>${s.speed} m/s</td>
          <td><b>${(s.likelihood * 100).toFixed(1)}%</b></td>
        `;
        tbody.appendChild(tr);
      });
    }

    function updateChart(step) {
      const activeSteps = simulationSteps.slice(0, step + 1);
      const labels = activeSteps.map((_, i) => `E${i + 1}`);
      const identities = [...new Set(simulationSteps.map(s => s.identityId))];

      const datasets = identities.map((id, idx) => {
        const colors = ['#22c55e', '#38bdf8', '#f59e0b', '#ec4899', '#8b5cf6'];
        const data = activeSteps.map(s => s.identityId === id ? s.likelihood : null);
        return {
          label: `Individual #${id}`,
          data: data,
          borderColor: colors[idx % colors.length],
          backgroundColor: colors[idx % colors.length],
          spanGaps: true,
          tension: 0.2
        };
      });

      probChart.data.labels = labels;
      probChart.data.datasets = datasets;
      probChart.update();
    }

    function togglePlay() {
      const btn = document.getElementById('playBtn');
      if (isPlaying) {
        clearInterval(timer);
        isPlaying = false;
        btn.innerText = '▶ Start Sequence';
      } else {
        if (currentStep >= simulationSteps.length - 1) setStep(0);
        const speed = parseInt(document.getElementById('speedSelect').value);
        isPlaying = true;
        btn.innerText = '⏸ Pause';
        timer = setInterval(() => {
          if (currentStep < simulationSteps.length - 1) {
            setStep(currentStep + 1);
          } else {
            togglePlay();
          }
        }, speed);
      }
    }

    function switchTab(tabId, btn) {
      document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
    }

    window.onload = initUI;
  </script>
</body>
</html>
"#);

    let mut file =
        File::create(output_path).expect("Unable to create HTML report file");
    file.write_all(html.as_bytes())
        .expect("Unable to write report file");
    println!(
        "📊 Multi-camera surveillance visualizer exported to: {}",
        output_path
    );
}
