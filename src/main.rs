//! Finds the shortest text (by character length) that includes all pairs of non-whitespace characters from a source text using only words from that source text.

use clap::Parser;
use dyad_reducer::{
    error::{ErrorKind, SolverError},
    solvers::greedy_most_valuable_word,
    text_model::Model,
};
use std::{
    fmt::{Debug, Formatter},
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    time::Instant,
};

/// An error returned by main().
///
/// The Debug implementation is equivalent to Display because returning an error from main prints the debug format.
struct LetterPairsError {
    message: String,
}

impl Debug for LetterPairsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl<E: std::error::Error> From<E> for LetterPairsError {
    fn from(error: E) -> Self {
        LetterPairsError {
            message: error.to_string(),
        }
    }
}

#[derive(Parser)]
#[command(version, about)]
struct Arguments {
    /// The path to the file to read.
    path: PathBuf,

    /// Path to the file in which solutions will be written. Creates a new file or truncates an existing one.
    #[arg(short)]
    output_path: Option<PathBuf>,

    /// Output statistics about the text and solution(s).
    #[arg(short, long)]
    statistics: bool,

    /// Perform a validation step to ensure the solution is valid.
    #[arg(short, long)]
    validate: bool,
}

/// Opens a file path for reading.
///
/// Returns a buffered reader on the contents of the file or an error if the file could not be opened.
fn open_file(path: &PathBuf) -> Result<BufReader<File>, SolverError> {
    match File::open(path) {
        Ok(file) => Ok(BufReader::new(file)),
        Err(error) => Err(SolverError::new(
            format!("Could not open file: {}.", path.display()),
            ErrorKind::InvalidFileError,
            Some(Box::new(error)),
        )),
    }
}

fn main() -> Result<(), LetterPairsError> {
    let mut output = String::new();
    let args = Arguments::parse();
    let output_file = match args.output_path.as_ref() {
        Some(path) => Some(File::create(path)?),
        None => None,
    };

    let model_start = Instant::now();
    let reader = open_file(&args.path)?;
    let model = Model::new(reader)?;
    let model_end = Instant::now();
    let solution = greedy_most_valuable_word(&model)?;
    let solver_end = Instant::now();

    if args.statistics {
        output.push_str(&format!(
            "File {} stats:\nNumber of unique character pairs: {}\nNumber of unique words: {}\nModel creation took {} seconds.\n\n",
            args.path.display(),
            model.pairs().len(),
            model.words().len(),
	    model_end.duration_since(model_start).as_secs_f32(),
        ));
        output.push_str(&format!(
            "Solution Stats:\nSolution calculation took {} seconds.\nNumber of unique character pairs: {}\nNumber of words: {}\nTotal length: {}\n",
	    solver_end.duration_since(model_end).as_secs_f32(),
	    solution.pairs().len(),
	    solution.words().len(),
	    solution.len(),
        ));
    };

    if args.validate {
        if !solution.is_valid(&model) {
            return Err(LetterPairsError {
                message: String::from(
                    "The greedy most valuable word algorithm produced an invalid solution.",
                ),
            });
        };
        output.push_str(&String::from("Solution validated successfully.\n"));
    }

    match output_file {
        Some(mut f) => {
            f.write_all(solution.to_string().as_bytes())?;
        }
        None => {
            output.push_str(&format!("\nSolution:\n{}", solution));
        }
    };
    println!("{}", output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_pair_error_from() {
        let message = String::from("cats are great.");
        let error = SolverError::new(message.clone(), ErrorKind::ModelConsistencyError, None);
        assert_eq!(message, format!("{:?}", LetterPairsError::from(error)));
    }

    #[test]
    fn letter_pair_error_format() {
        let message = String::from("cats are great.");
        let error = LetterPairsError {
            message: message.clone(),
        };
        assert_eq!(message, format!("{:?}", error));
    }

    #[test]
    fn verify_arguments() {
        use clap::CommandFactory;
        Arguments::command().debug_assert()
    }
}
