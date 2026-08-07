//! # LI-ESKG factor layer
//!
//! This crate provides abstract traits for local factor functions
//! $\phi_i(Z_\phi)$ and factor graph compilation interfaces. It contains no
//! domain-specific spatial, temporal, or semantic assumptions.

#![deny(unsafe_code)]

pub mod provider;

pub use provider::{
    CandidateBuffer, FactorBuffer, FactorProvider, FactorTable,
    ProviderContext, ProviderError,
};
