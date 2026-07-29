//! Graph operation execution and persistence dispatching component.

use alloc::vec::Vec;

use li_model::operations::GraphOperation;

/// Abstract trait representing a sink capable of persisting graph operations.
pub trait ExecutionSink<P, E, S> {
    /// Error type produced when execution fails.
    type Error;

    /// Persists a batch of graph operations to the underlying storage or
    /// backend.
    ///
    /// # Arguments
    ///
    /// * `operations` - Slice of graph operations to execute.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful execution or `Err(Self::Error)` on
    /// failure.
    fn execute_batch(
        &mut self,
        operations: &[GraphOperation<P, E, S>],
    ) -> Result<(), Self::Error>;
}

/// Orchestrates execution and routing of graph mutation operations to storage
/// sinks.
#[derive(Debug)]
pub struct OperationExecutor<S> {
    sink: S,
}

impl<S> OperationExecutor<S> {
    /// Creates a new `OperationExecutor` wrapping a specific storage sink.
    ///
    /// # Arguments
    ///
    /// * `sink` - The destination sink for executed operations.
    pub fn new(sink: S) -> Self {
        Self { sink }
    }

    /// Dispatches a sequence of operations to the underlying execution sink.
    ///
    /// # Arguments
    ///
    /// * `operations` - Vector of operations to commit.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all operations are executed successfully.
    pub fn commit<P, E, St, Err>(
        &mut self,
        operations: Vec<GraphOperation<P, E, St>>,
    ) -> Result<(), Err>
    where
        S: ExecutionSink<P, E, St, Error = Err>,
    {
        if operations.is_empty() {
            return Ok(());
        }
        self.sink.execute_batch(&operations)
    }

    /// Dispatches borrowed operations without consuming their backing buffer.
    ///
    /// This entry point allows real-time callers to retain and reuse a
    /// preallocated operation buffer between commits.
    ///
    /// # Arguments
    ///
    /// * `operations` - Borrowed operations to commit.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all operations are executed successfully.
    pub fn commit_slice<P, E, St, Err>(
        &mut self,
        operations: &[GraphOperation<P, E, St>],
    ) -> Result<(), Err>
    where
        S: ExecutionSink<P, E, St, Error = Err>,
    {
        if operations.is_empty() {
            return Ok(());
        }
        self.sink.execute_batch(operations)
    }

    /// Returns an immutable reference to the underlying execution sink.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Returns a mutable reference to the underlying execution sink.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSink {
        executed_count: usize,
    }

    impl<P, E, S> ExecutionSink<P, E, S> for MockSink {
        type Error = ();

        fn execute_batch(
            &mut self,
            operations: &[GraphOperation<P, E, S>],
        ) -> Result<(), Self::Error> {
            self.executed_count += operations.len();
            Ok(())
        }
    }

    #[test]
    fn test_executor_commit_empty() {
        let sink = MockSink { executed_count: 0 };
        let mut executor = OperationExecutor::new(sink);
        let ops: Vec<GraphOperation<(), (), ()>> = Vec::new();

        assert!(executor.commit(ops).is_ok());
        assert_eq!(executor.sink.executed_count, 0);
    }

    #[test]
    fn test_executor_commit_slice_preserves_reusable_buffer() {
        let sink = MockSink { executed_count: 0 };
        let mut executor = OperationExecutor::new(sink);
        let operations =
            Vec::from([GraphOperation::<(), (), ()>::MergeIdentities {
                target: li_core::ids::IdentityId(1),
                duplicate: li_core::ids::IdentityId(2),
            }]);
        let original_capacity = operations.capacity();

        assert!(executor.commit_slice(&operations).is_ok());
        assert_eq!(executor.sink.executed_count, 1);
        assert_eq!(operations.capacity(), original_capacity);
    }
}
