//! Finds the shortest text (by character length) that includes all pairs of non-whitespace characters from a source text using only words from that source text.

#![warn(missing_docs)]
mod consts;
pub mod error;
pub mod solution;
pub mod solvers;
pub mod stats;
#[cfg(test)]
mod test_reader;
pub mod text_model;
