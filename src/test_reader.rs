use std::io;

pub struct TestReader {
    reader: Vec<u8>,
    error: Option<(io::ErrorKind, String)>,
}

impl TestReader {
    pub fn new(text: &str, error: Option<(io::ErrorKind, String)>) -> TestReader {
        TestReader {
            reader: Vec::from(text.as_bytes()),
            error,
        }
    }
}

impl io::Read for TestReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.error.is_none() {
            let bytes_read = self.reader.as_slice().read(buf)?;
            self.reader = self.reader.split_off(bytes_read);
            Ok(bytes_read)
        } else {
            Err(io::Error::new(
                self.error.clone().unwrap().0,
                self.error.clone().unwrap().1,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use io::Read;

    #[test]
    fn read_empty() {
        let mut reader = TestReader::new("", None);
        let mut buf = Vec::<u8>::new();
        let num_bytes = reader.read(&mut buf).unwrap();
        assert_eq!(num_bytes, 0);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn read_error() {
        let kind = io::ErrorKind::Other;
        let message = String::from("test_error");
        let mut reader = TestReader::new("some text you might", Some((kind, message.clone())));
        let mut buf = Vec::<u8>::new();
        let error = reader.read(&mut buf).unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn read_stuff() {
        let message = "cat-wolf abs";
        let mut reader = TestReader::new(message, None);
        let mut buf = [0u8; 12];
        let mut num_bytes = reader.read(&mut buf).unwrap();
        assert_eq!(num_bytes, message.len());
        assert_eq!(buf.as_slice(), message.as_bytes());
        num_bytes = reader.read(&mut buf).unwrap();
        assert_eq!(num_bytes, 0);
    }
}
