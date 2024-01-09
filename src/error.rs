//! Errors conditions returned by the program.
use std::{
    cmp::PartialEq,
    error::Error,
    fmt::{self, Display},
};

/// An error
#[derive(Debug)]
pub struct GenericError<T: fmt::Debug + PartialEq> {
    message: String,
    kind: T,
    source: Option<Box<dyn Error>>,
}

impl<T: fmt::Debug + PartialEq> GenericError<T> {
    /// Builds a GenericError
    pub fn new(message: String, kind: T, source: Option<Box<dyn Error>>) -> Self {
        GenericError {
            message,
            kind,
            source,
        }
    }

    /// Returns what kind of error this error is.
    pub fn kind(&self) -> &T {
        &self.kind
    }
}

impl<T: fmt::Debug + PartialEq> Display for GenericError<T> {
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

impl<T: fmt::Debug + PartialEq> Error for GenericError<T> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref()
    }
}

impl<T: fmt::Debug + PartialEq> PartialEq for GenericError<T> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.message == other.message
            && if let Some(source) = self.source.as_ref() {
                other.source.as_ref().map_or(false, |other_source| {
                    source.to_string() == other_source.to_string()
                })
            } else {
                other.source.is_none()
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{error::Error, io};

    type TestError = GenericError<bool>;

    #[test]
    fn display_no_source() {
        let message = String::from("Error: cats are great");
        let error = TestError {
            message: message.clone(),
            kind: true,
            source: None,
        };
        assert_eq!(error.to_string(), message)
    }

    #[test]
    fn display_a_source() {
        let submessage = String::from("Suberror: cats are great");
        let suberror = TestError {
            message: submessage.clone(),
            kind: false,
            source: None,
        };
        let message = String::from("Supererror: cats have abs");
        let error = TestError {
            message: message.clone(),
            kind: true,
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
        let error = TestError {
            message: String::from("external"),
            kind: true,
            source: Some(Box::new(internal_error)),
        };
        let error_source = error.source.expect("error source should exist");
        assert_eq!(error_source.to_string(), copy.to_string());
    }

    #[test]
    fn source_none() {
        let error = TestError {
            message: String::from("external"),
            kind: false,
            source: None,
        };
        assert!(error.source().is_none());
    }

    fn create_sub_error() -> TestError {
        TestError {
            message: String::from("sub"),
            kind: true,
            source: None,
        }
    }

    fn create_test_error() -> TestError {
        TestError {
            message: String::from("test"),
            kind: false,
            source: Some(Box::new(create_sub_error())),
        }
    }

    #[test]
    fn partial_eq_equal() {
        let e1 = create_test_error();
        let e2 = create_test_error();
        assert_eq!(e1, e2);
    }

    #[test]
    fn partial_eq_diff_message() {
        let e1 = create_test_error();
        let mut e2 = create_test_error();
        e2.message = String::from("silly");
        assert_ne!(e1, e2);
    }

    #[test]
    fn partial_eq_diff_kind() {
        let e1 = create_test_error();
        let mut e2 = create_test_error();
        e2.kind = !e2.kind;
        assert_ne!(e1, e2);
    }

    #[test]
    fn partial_eq_diff_sub() {
        let e1 = create_test_error();
        let mut e2 = create_test_error();
        let mut sub = create_sub_error();
        sub.message.pop();
        e2.source = Some(Box::new(sub));
        assert_ne!(e1, e2);
    }

    #[test]
    fn partial_eq_other_has_sub() {
        let mut e1 = create_test_error();
        e1.source = None;
        let e2 = create_test_error();
        assert_ne!(e1, e2);
    }

    #[test]
    fn partial_eq_other_missing_sub() {
        let e1 = create_test_error();
        let mut e2 = create_test_error();
        e2.source = None;
        assert_ne!(e1, e2);
    }
}
