//! Errors conditions returned by the program.
use std::{
    error::Error,
    fmt::{self, Display},
};

/// An error returned by a solver.
#[derive(Debug)]
pub struct SolverError {
    message: String,
    kind: ErrorKind,
    source: Option<Box<dyn Error>>,
}

impl SolverError {
    /// Builds a SolverError
    pub fn new(message: String, kind: ErrorKind, source: Option<Box<dyn Error>>) -> Self {
        SolverError {
            message,
            kind,
            source,
        }
    }

    /// Returns what kind of error this error is.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let source = match &self.source {
            Some(x) => {
                format!("\nThe underlying cause of this error is: {}", x)
            }
            None => String::from(""),
        };
        write!(f, "{}{}", self.message, source)
    }
}

impl Error for SolverError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            Some(x) => Some(&**x),
            None => None,
        }
    }
}

/// An enum specifying the category of error.
#[derive(Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// The specified file was invalid.
    InvalidFileError,
    /// The text Model is inconsistent.
    ///
    /// e.g. the solution is not complete, but there are no more unchosen words to choose from.
    ModelConsistencyError,
    /// Could not read from file.
    FileReadingError,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn display_no_source() {
        let message = String::from("Error: cats are great");
        let error = SolverError {
            message: message.clone(),
            kind: ErrorKind::InvalidFileError,
            source: None,
        };
        assert_eq!(error.to_string(), message)
    }

    #[test]
    fn display_a_source() {
        let submessage = String::from("Suberror: cats are great");
        let suberror = SolverError {
            message: submessage.clone(),
            kind: ErrorKind::InvalidFileError,
            source: None,
        };
        let message = String::from("Supererror: cats have abs");
        let error = SolverError {
            message: message.clone(),
            kind: ErrorKind::ModelConsistencyError,
            source: Some(Box::new(suberror)),
        };
        assert_eq!(
            error.to_string(),
            format!(
                "{}\nThe underlying cause of this error is: {}",
                message, submessage
            )
        );
    }

    #[test]
    fn source_some() {
        let internal_error = io::Error::new(io::ErrorKind::Other, "internal");
        let copy = io::Error::new(io::ErrorKind::Other, "internal");
        let error = SolverError {
            message: String::from("external"),
            kind: ErrorKind::InvalidFileError,
            source: Some(Box::new(internal_error)),
        };
        let error_source = error.source.expect("error source should exist");
        assert_eq!(error_source.to_string(), copy.to_string());
    }

    #[test]
    fn source_none() {
        let error = SolverError {
            message: String::from("external"),
            kind: ErrorKind::InvalidFileError,
            source: None,
        };
        assert!(error.source().is_none());
    }
}
