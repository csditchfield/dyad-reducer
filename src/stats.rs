//! Utilities for calculating statistics about Solution generation.
use crate::{error::SolverError, solution::Solution, text_model::Model};
use std::{
    fmt::{self, Display, Formatter},
    time::{Duration, Instant},
};

struct Timing {
    result: Result<Solution, SolverError>,
    duration: Duration,
}

struct SolutionStats {
    mean_length: f64,
    median_length: f64,
    min_length: usize,
    max_length: usize,
    mean_count: f64,
    median_count: f64,
    min_count: usize,
    max_count: usize,
    mean_duration: f64,
    median_duration: f64,
    min_duration: usize,
    max_duration: usize,
}

/// A struct representing the statistics of a Sample of Solutions.
pub struct Stats {
    successes: usize,
    failures: usize,
    solution_stats: Option<SolutionStats>,
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut output = format!(
            "Solution generation statistics:\nTotal Runs: {}\nSuccesses: {}\nFailures: {}",
            self.runs(),
            self.successes(),
            self.failures()
        );
        if let Some(stats) = &self.solution_stats {
            output.push_str(&format!(
		"\nmean_length: {}\nmedian_length: {}\nmin_length: {}\nmax_length: {}\nmean_count: {}\nmedian_count: {}\nmin_count: {}\nmax_count: {}\nmean_duration: {}\nmedian_duration: {}\nmin_duration: {}\nmax_duration: {}",
		stats.mean_length,
		stats.median_length,
		stats.min_length,
		stats.max_length,
		stats.mean_count,
		stats.median_count,
		stats.min_count,
		stats.max_count,
		stats.mean_duration,
		stats.median_duration,
		stats.min_duration,
		stats.max_duration,
	    ));
        };
        write!(f, "{}", output)
    }
}

impl Stats {
    /// Returns the total number of runs.
    pub fn runs(&self) -> usize {
        self.successes() + self.failures()
    }

    /// Returns the number of runs that created a Solution successfuly.
    pub fn successes(&self) -> usize {
        self.successes
    }

    /// Returns the number of runs that failed to produce a solution.
    pub fn failures(&self) -> usize {
        self.failures
    }

    /// Returns the mean character length of the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn mean_length(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.mean_length)
    }

    /// Returns the median character length of the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn median_length(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.median_length)
    }

    /// Returns the character length of the shortest solution.
    ///
    /// Returns None if there are no Solutions.
    pub fn min_length(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.min_length)
    }

    /// Returns the character length of the longest solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn max_length(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.max_length)
    }

    /// Returns the mean word count of the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn mean_count(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.mean_count)
    }

    /// Returns the median word count of the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn median_count(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.median_count)
    }

    /// Returns the word count of the solution with the fewest words.
    ///
    /// Returns None if there are no Solutions.
    pub fn min_count(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.min_count)
    }

    /// Returns the word count of the solution with the most words.
    ///
    /// Returns None if there are no Solutions.
    pub fn max_count(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.max_count)
    }

    /// Returns the mean computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn mean_duration(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.mean_duration)
    }

    /// Returns the median computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn median_duration(&self) -> Option<f64> {
        self.solution_stats.as_ref().map(|s| s.median_duration)
    }

    /// Returns the minimum computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn min_duration(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.min_duration)
    }

    /// Returns the maximum computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn max_duration(&self) -> Option<usize> {
        self.solution_stats.as_ref().map(|s| s.max_duration)
    }
}

/// A collection of Solutions combined with the time it took to create them.
pub struct Sample {
    timings: Vec<Timing>,
}

impl Sample {
    /// Create a new Sample using a solver function on a Model 0 or more times.
    pub fn new(
        solver: &dyn Fn(&Model) -> Result<Solution, SolverError>,
        model: &Model,
        n: &usize,
    ) -> Self {
        let mut timings = Vec::new();
        for _ in 0..*n {
            let start = Instant::now();
            let result = solver(model);
            let duration = start.elapsed();
            timings.push(Timing { result, duration });
        }
        Self { timings }
    }

    /// Calculate the statistics of this Sample.
    pub fn calculate_stats(&self) -> Stats {
        Stats {
            successes: 0,
            failures: 0,
            solution_stats: None,
        }
    }

    /// Validates all solutions in the sample and returns the number of solutions that failed.
    pub fn validate(&self, model: &Model) -> usize {
        todo!()
    }

    /// Find the shortest Solution in the Sample.
    ///
    /// Returns None if there are no Solutions.
    pub fn best_solution(&self) -> Option<&Solution> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod stats {
        use super::*;

        fn build_test_solution_stats() -> Option<SolutionStats> {
            Some(SolutionStats {
                mean_length: 1.0,
                median_length: 2.0,
                min_length: 3,
                max_length: 4,
                mean_count: 5.0,
                median_count: 6.0,
                min_count: 7,
                max_count: 8,
                mean_duration: 9.0,
                median_duration: 10.0,
                min_duration: 11,
                max_duration: 12,
            })
        }

        #[test]
        fn runs_none() {
            let stats = Stats {
                successes: 0,
                failures: 0,
                solution_stats: None,
            };
            assert_eq!(stats.runs(), 0);
        }

        #[test]
        fn runs_successes() {
            let stats = Stats {
                successes: 14,
                failures: 0,
                solution_stats: None,
            };
            assert_eq!(stats.runs(), 14);
        }

        #[test]
        fn runs_failures() {
            let stats = Stats {
                successes: 0,
                failures: 14,
                solution_stats: None,
            };
            assert_eq!(stats.runs(), 14);
        }

        #[test]
        fn runs_both() {
            let stats = Stats {
                successes: 14,
                failures: 14,
                solution_stats: None,
            };
            assert_eq!(stats.runs(), 28);
        }

        #[test]
        fn mean_length_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.mean_length().is_none());
        }

        #[test]
        fn mean_length_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .mean_length()
                    .expect("Test stats should have solution stat block."),
                1.0
            );
        }

        #[test]
        fn median_length_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.median_length().is_none());
        }

        #[test]
        fn median_length_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .median_length()
                    .expect("Test stats should have solution stat block."),
                2.0
            );
        }

        #[test]
        fn min_length_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.min_length().is_none());
        }

        #[test]
        fn min_length_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .min_length()
                    .expect("Test stats should have solution stat block."),
                3
            );
        }

        #[test]
        fn max_length_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.max_length().is_none());
        }

        #[test]
        fn max_length_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .max_length()
                    .expect("Test stats should have solution stat block."),
                4
            );
        }

        #[test]
        fn mean_count_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.mean_count().is_none());
        }

        #[test]
        fn mean_count_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .mean_count()
                    .expect("Test stats should have solution stat block."),
                5.0
            );
        }

        #[test]
        fn median_count_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.median_count().is_none());
        }

        #[test]
        fn median_count_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .median_count()
                    .expect("Test stats should have solution stat block."),
                6.0
            );
        }

        #[test]
        fn min_count_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.min_count().is_none());
        }

        #[test]
        fn min_count_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .min_count()
                    .expect("Test stats should have solution stat block."),
                7
            );
        }

        #[test]
        fn max_count_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.max_count().is_none());
        }

        #[test]
        fn max_count_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .max_count()
                    .expect("Test stats should have solution stat block."),
                8
            );
        }

        #[test]
        fn mean_duration_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.mean_duration().is_none());
        }

        #[test]
        fn mean_duration_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .mean_duration()
                    .expect("Test stats should have solution stat block."),
                9.0
            );
        }

        #[test]
        fn median_duration_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.median_duration().is_none());
        }

        #[test]
        fn median_duration_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .median_duration()
                    .expect("Test stats should have solution stat block."),
                10.0
            );
        }

        #[test]
        fn min_duration_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.min_duration().is_none());
        }

        #[test]
        fn min_duration_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .min_duration()
                    .expect("Test stats should have solution stat block."),
                11
            );
        }

        #[test]
        fn max_duration_none() {
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            };
            assert!(stats.max_duration().is_none());
        }

        #[test]
        fn max_duration_some() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 32,
                failures: 0,
                solution_stats,
            };
            assert_eq!(
                stats
                    .max_duration()
                    .expect("Test stats should have solution stat block."),
                12
            );
        }

        #[test]
        fn display_empty() {
            let stats = Stats {
                successes: 32,
                failures: 32,
                solution_stats: None,
            };
            assert_eq!(
                stats.to_string(),
                "Solution generation statistics:\nTotal Runs: 64\nSuccesses: 32\nFailures: 32"
            )
        }

        #[test]
        fn display_stats() {
            let solution_stats = build_test_solution_stats();
            let stats = Stats {
                successes: 24,
                failures: 8,
                solution_stats,
            };
            let expected = "Solution generation statistics:\nTotal Runs: 32\nSuccesses: 24\nFailures: 8\nmean_length: 1\nmedian_length: 2\nmin_length: 3\nmax_length: 4\nmean_count: 5\nmedian_count: 6\nmin_count: 7\nmax_count: 8\nmean_duration: 9\nmedian_duration: 10\nmin_duration: 11\nmax_duration: 12";
            assert_eq!(stats.to_string(), expected)
        }
    }

    mod sample_tests {
        use super::*;

        #[test]
        fn empty() {
            todo!();
        }

        #[test]
        fn one_run() {
            todo!();
        }

        #[test]
        fn multiple_runs() {
            todo!();
        }

        #[test]
        fn error() {
            todo!();
        }

        #[test]
        fn error_multiple() {
            todo!();
        }
    }
}
