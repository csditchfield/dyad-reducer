//! Finds the shortest text (by character length) that includes all pairs of non-whitespace characters from a source text using only words from that source text.

use clap::Parser;
use dyad_reducer::{solvers::greedy_most_valuable_word, stats::Sample, text_model::Model};
use std::{
    fmt::{Debug, Formatter},
    fs::File,
    io::{self, BufReader, Write},
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

    /// Calculate the solution REPEAT times and output the shortest solution.
    #[arg(short, long, default_value_t = 1)]
    repeat: usize,
}

/// Opens a file path for reading.
///
/// Returns a buffered reader on the contents of the file or an error if the file could not be opened.
fn open_file(path: &PathBuf) -> Result<BufReader<File>, io::Error> {
    File::open(path).map(BufReader::new)
}

fn main() -> Result<(), LetterPairsError> {
    let mut output = String::new();
    let args = Arguments::parse();
    let output_file = match args.output_path.as_ref() {
        Some(path) => Some(File::create(path)?),
        None => None,
    };

    let reader = open_file(&args.path)?;
    let model_start = Instant::now();
    let model = Model::new(reader)?;
    let model_end = Instant::now();
    let sample = Sample::new(greedy_most_valuable_word, &model, args.repeat);

    if args.validate {
        let validate_start = Instant::now();
        let failures = sample.validate_solutions(&model);
        let validation_duration = validate_start.elapsed();

        if failures > 0 {
            return Err(LetterPairsError {
                message: format!(
                    "The greedy most valuable word algorithm produced {} invalid solution(s).",
                    failures,
                ),
            });
        };
        output.push_str(&String::from("All solutions validated successfully.\n"));

        if args.statistics {
            output.push_str(&format!(
                "Validation performed in {} seconds.\n",
                validation_duration.as_secs_f64()
            ));
        }
        output.push('\n');
    }

    if args.statistics {
        output.push_str(&format!(
            "File {} stats:\nNumber of unique character pairs: {}\nNumber of unique words: {}\nModel creation took {} seconds.\n\n",
            args.path.display(),
            model.pairs().len(),
            model.words().len(),
	    model_end.duration_since(model_start).as_secs_f32(),
        ));

        let stats = sample.calculate_stats();
        if stats.runs() > 1 {
            output.push_str(&(stats.to_string() + "\n\n"));
        }
    };

    if let Some((best, time_of_best)) = sample.best_solution() {
        if args.statistics {
            output.push_str( &format!(
		"Best solution stats:\nLength: {}\nNumber of unique words: {}\nNumber of unique character pairs: {}\nDuration: {}\n\n",
		best.len(),
		best.words().len(),
		best.pairs().len(),
		time_of_best.as_secs_f64(),
	    ))
        }

        match output_file {
            Some(mut f) => {
                f.write_all(best.to_string().as_bytes())?;
            }
            None => {
                output.push_str(&format!("Solution:\n{}", best));
            }
        };
    };
    println!("{}", output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use dyad_reducer::solvers;

    #[test]
    fn letter_pair_error_from() {
        let message = String::from("cats are great.");
        let error = solvers::SolverError::new(
            message.clone(),
            solvers::ErrorKind::ModelConsistencyError,
            None,
        );
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
