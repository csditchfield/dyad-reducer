//! Finds the shortest text (by character length) that includes all pairs of non-whitespace characters from a source text using only words from that source text.

#![warn(missing_docs)]
pub mod error;
pub mod solution;
pub mod solvers;
pub mod text_model;

#[cfg(test)]
mod test_reader;
