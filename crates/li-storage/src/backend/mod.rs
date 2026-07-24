//! Backend drivers for no-std memory storage and std RocksDB integration.

pub mod memory;
pub use memory::MemoryKvBackend;

#[cfg(feature = "std")]
pub mod rocksdb;
#[cfg(feature = "std")]
pub use self::rocksdb::RocksDbBackend;
