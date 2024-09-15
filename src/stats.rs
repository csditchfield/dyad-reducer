//! Utilities for calculating statistics about Solution generation.
use crate::{
    consts::BILLION_SQUARED,
    error::GenericError,
    solution::Solution,
    solvers::{Solver, SolverError},
    text_model::Model,
};
use std::{
    fmt::{self, Display, Formatter},
    time::{Duration, Instant},
};

/// An error raised by the stats library.
pub type StatsError = GenericError<ErrorKind>;

/// The kind of error.
#[derive(Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// The calculation could not complete do to overflow.
    OverflowError(OverflowLocation),
    /// The calculation could not be performed because we are missing a required value.
    MissingValue,
}

/// The kind of error.
#[derive(Debug, Eq, PartialEq)]
pub enum OverflowLocation {
    /// Mean calculation overflowed during summation.
    MeanSummation,
    /// Variance calculation overflowed while calculating the square of a deviation.
    DeviationSquaring,
    ///Variance calculation overflowed while summing squared deviations.
    SquareDeviationSummation,
}

#[derive(Debug)]
/// A type that encapsulates a `solution::Solution` and the time it took to calculate.
struct Timing {
    result: Result<Solution, SolverError>,
    duration: Duration,
}

impl PartialEq for Timing {
    fn eq(&self, other: &Self) -> bool {
        return if self.duration != other.duration
            || (self.result.is_err() && other.result.is_ok())
            || (self.result.is_ok() && other.result.is_err())
        {
            false
        } else if let (Ok(s1), Ok(s2)) = (self.result.as_ref(), other.result.as_ref()) {
            s1 == s2
        } else {
            true
        };
    }
}

#[derive(Debug, PartialEq)]
struct SolutionStats {
    mean_length: Result<f64, StatsError>,
    median_length: f64,
    min_length: usize,
    max_length: usize,
    uncorrected_length_variance: Result<f64, StatsError>,
    uncorrected_length_standard_deviation: Result<f64, StatsError>,
    mean_count: Result<f64, StatsError>,
    median_count: f64,
    min_count: usize,
    max_count: usize,
    uncorrected_count_variance: Result<f64, StatsError>,
    uncorrected_count_standard_deviation: Result<f64, StatsError>,
    mean_duration: Result<Duration, StatsError>,
    median_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    uncorrected_duration_variance: Result<u128, StatsError>, // In nanoseconds squared
    uncorrected_duration_standard_deviation: Result<f64, StatsError>, // In seconds
}

/// A struct representing the statistics of a Sample of Solutions.
#[derive(Debug, PartialEq)]
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
            match stats.mean_length.as_ref() {
                Ok(mean) => output.push_str(&format!("\nmean length: {}", mean)),
                Err(err) => output.push_str(&format!(
                    "\ncould not calculate mean length because {}",
                    err
                )),
            };
            output.push_str(&format!(
                "\nmedian_length: {}\nmin_length: {}\nmax_length: {}",
                stats.median_length, stats.min_length, stats.max_length,
            ));
            match stats.uncorrected_length_standard_deviation.as_ref() {
                Ok(std_dev) => {
                    output.push_str(&format!("\nuncorrected standard deviation: {}", std_dev))
                }
                Err(err) => {
                    output.push_str(&format!(
                        "\ncould not calculate uncorrected standard deviation because {}",
                        err
                    ));
                }
            };

            match stats.mean_count.as_ref() {
                Ok(mean) => output.push_str(&format!("\nmean count: {}", mean)),
                Err(err) => {
                    output.push_str(&format!("\ncould not calculate mean count because {}", err))
                }
            };
            output.push_str(&format!(
                "\nmedian_count: {}\nmin_count: {}\nmax_count: {}",
                stats.median_count, stats.min_count, stats.max_count,
            ));
            match stats.uncorrected_count_standard_deviation.as_ref() {
                Ok(std_dev) => {
                    output.push_str(&format!("\nuncorrected standard deviation: {}", std_dev))
                }
                Err(err) => output.push_str(&format!(
                    "\ncould not calculate uncorrected standard deviation because {}",
                    err
                )),
            };

            match stats.mean_duration.as_ref() {
                Ok(mean) => {
                    output.push_str(&format!("\nmean compute time: {}s", mean.as_secs_f64()))
                }
                Err(err) => output.push_str(&format!(
                    "\ncould not calculate mean compute time because {}",
                    err
                )),
            };
            output.push_str(&format!(
                "\nmedian compute time: {}s\nmin compute time: {}s\nmax compute time: {}s",
                stats.median_duration.as_secs_f64(),
                stats.min_duration.as_secs_f64(),
                stats.max_duration.as_secs_f64()
            ));
            match stats.uncorrected_duration_standard_deviation.as_ref() {
                Ok(std_dev) => {
                    output.push_str(&format!("\nuncorrected standard deviation: {}s", std_dev))
                }
                Err(err) => output.push_str(&format!(
                    "\ncould not calculate uncorrected standard deviation because {}",
                    err
                )),
            };
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
    pub fn mean_length(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats.as_ref().map(|s| &s.mean_length)
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

    /// Returns the uncorrected sample variance of the solutions' lengths.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_length_variance(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_length_variance)
    }

    /// Returns the uncorrected sample standard deviation of the solutions' lengths.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_length_standard_deviation(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_length_standard_deviation)
    }

    /// Returns the mean word count of the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn mean_count(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats.as_ref().map(|s| &s.mean_count)
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

    /// Returns the uncorrected sample variance of the solutions' word counts.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_count_variance(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_count_variance)
    }

    /// Returns the uncorrected sample standard deviation of the solutions' word counts.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_count_standard_deviation(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_count_standard_deviation)
    }

    /// Returns the mean computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn mean_duration(&self) -> Option<&Result<Duration, StatsError>> {
        self.solution_stats.as_ref().map(|s| &s.mean_duration)
    }

    /// Returns the median computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn median_duration(&self) -> Option<Duration> {
        self.solution_stats.as_ref().map(|s| s.median_duration)
    }

    /// Returns the minimum computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    pub fn min_duration(&self) -> Option<Duration> {
        self.solution_stats.as_ref().map(|s| s.min_duration)
    }

    /// Returns the maximum computation time taken to create the solutions.
    ///
    /// Returns None if there are no Solutions.
    /// If there are multiple solutions of the same length, one is returned arbitrarily.
    pub fn max_duration(&self) -> Option<Duration> {
        self.solution_stats.as_ref().map(|s| s.max_duration)
    }

    /// Returns the uncorrected sample variance of the solutions' computation time.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_duration_variance(&self) -> Option<&Result<u128, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_duration_variance)
    }

    /// Returns the uncorrected sample standard deviation of the solutions' computation time.
    ///
    /// Returns None if there are no Solutions.
    pub fn uncorrected_duration_standard_deviation(&self) -> Option<&Result<f64, StatsError>> {
        self.solution_stats
            .as_ref()
            .map(|s| &s.uncorrected_duration_standard_deviation)
    }
}

/// A collection of Solutions combined with the time it took to create them.
pub struct Sample {
    timings: Vec<Timing>,
}

impl Sample {
    /// Create a new Sample using a solver function on a Model 0 or more times.
    pub fn new<S>(solver: S, model: &Model, n: usize) -> Self
    where
        S: Solver,
    {
        let mut timings = Vec::new();
        for _ in 0..n {
            let start = Instant::now();
            let result = solver(model);
            let duration = start.elapsed();
            timings.push(Timing { result, duration });
        }
        Self { timings }
    }

    fn calculate_mean_usize(nums: &[usize]) -> Option<Result<f64, StatsError>> {
        if !nums.is_empty() {
            Some(
                nums.iter()
                    .try_fold(usize::MIN, |acc, &x| acc.checked_add(x))
                    .map(|m| m as f64 / nums.len() as f64)
                    .ok_or_else(|| {
                        StatsError::new(
                            String::from(
                                "summing of usizes overflowed while calculating mean value",
                            ),
                            ErrorKind::OverflowError(OverflowLocation::MeanSummation),
                            None,
                        )
                    }),
            )
        } else {
            None
        }
    }

    fn calculate_median_usize(nums: &mut [usize]) -> Option<f64> {
        nums.sort_unstable();

        if !nums.is_empty() {
            let mid = nums.len() / 2;
            if nums.len() % 2 == 0 {
                Some(
                    nums[mid - 1..mid + 1]
                        .iter()
                        .map(|x| *x as f64 / 2.0)
                        .sum::<f64>(),
                )
            } else {
                Some(nums[mid] as f64)
            }
        } else {
            None
        }
    }

    fn calculate_uncorrected_variance_usize(nums: &[usize], mean: f64) -> Option<f64> {
        if !nums.is_empty() {
            Some(nums.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / nums.len() as f64)
        } else {
            None
        }
    }

    fn calculate_mean_duration(durations: &[Duration]) -> Option<Result<Duration, StatsError>> {
        if !durations.is_empty() {
            Some(
                durations
                    .iter()
                    .try_fold(Duration::ZERO, |acc, &x| acc.checked_add(x))
                    .map(|s| s / durations.len() as u32)
                    .ok_or(StatsError::new(
                        String::from(
                            "summing of Durations overflowed while calculating mean value",
                        ),
                        ErrorKind::OverflowError(OverflowLocation::MeanSummation),
                        None,
                    )),
            ) // Don't worry about len > u32 because of stack overflow.
        } else {
            None
        }
    }

    fn calculate_median_duration(durations: &mut [Duration]) -> Option<Duration> {
        durations.sort_unstable();

        if !durations.is_empty() {
            let mid = durations.len() / 2;
            if durations.len() % 2 == 0 {
                Some(
                    durations[mid - 1..mid + 1]
                        .iter()
                        .map(|x| *x / 2)
                        .sum::<Duration>(),
                )
            } else {
                Some(durations[mid])
            }
        } else {
            None
        }
    }

    /// Calculates the uncorrected variance of a slice of Durations.
    fn calculate_uncorrected_variance_duration(
        durations: &[Duration],
        mean: &Duration,
    ) -> Option<Result<u128, StatsError>> {
        if !durations.is_empty() {
            let mut sum: u128 = 0;

            for duration in durations.iter() {
                let deviation = duration
                    .checked_sub(*mean)
                    .unwrap_or_else(|| *mean - *duration)
                    .as_nanos();
                if let Some(squared) = deviation.checked_pow(2) {
                    if let Some(s) = sum.checked_add(squared) {
                        sum = s;
                    } else {
                        return Some(Err(StatsError::new(
                            String::from("summing of squared Duration deviations overflowed"),
                            ErrorKind::OverflowError(OverflowLocation::SquareDeviationSummation),
                            None,
                        )));
                    };
                } else {
                    return Some(Err(StatsError::new(
                        String::from("squaring deviations overflowed"),
                        ErrorKind::OverflowError(OverflowLocation::DeviationSquaring),
                        None,
                    )));
                }
            }
            Some(Ok(sum / durations.len() as u128))
        } else {
            None
        }
    }

    fn nanos_squared_to_secs_squared(nanos_squared: u128) -> f64 {
        (nanos_squared / BILLION_SQUARED) as f64
            + ((nanos_squared % BILLION_SQUARED) as f64 / BILLION_SQUARED as f64)
    }

    /// Calculate the statistics of this Sample.
    pub fn calculate_stats(&self) -> Stats {
        let mut successes = 0;
        let mut failures = 0;
        let mut lengths = Vec::new();
        let mut word_counts = Vec::new();
        let mut durations = Vec::new();

        for timing in self.timings.iter() {
            if let Ok(solution) = timing.result.as_ref() {
                successes += 1;
                lengths.push(solution.len());
                word_counts.push(solution.words().len());
                durations.push(timing.duration);
            } else {
                failures += 1;
            };
        }

        lengths.sort_unstable();
        word_counts.sort_unstable();
        durations.sort_unstable();

        let solution_stats = if successes > 0 {
            let mean_length = Sample::calculate_mean_usize(&lengths)
                .expect("Lengths should have a mean when there are solutions.");
            let (uncorrected_length_variance, uncorrected_length_standard_deviation) =
                if let Ok(mean) = mean_length {
                    let variance = Sample::calculate_uncorrected_variance_usize(&lengths, mean)
                        .expect("Lengths should have a variance when there are solutions.");
                    let std_dev = variance.sqrt();
                    (Ok(variance), Ok(std_dev))
                } else {
                    (
                        Err(StatsError::new(
                            String::from("could not calculate length variance, mean not available"),
                            ErrorKind::MissingValue,
                            None,
                        )),
                        Err(StatsError::new(
                            String::from(
                                "could not calculate length standard deviation, mean not available",
                            ),
                            ErrorKind::MissingValue,
                            None,
                        )),
                    )
                };

            let mean_count = Sample::calculate_mean_usize(&word_counts)
                .expect("Word count should have a mean when there are solutions.");
            let (uncorrected_count_variance, uncorrected_count_standard_deviation) =
                if let Ok(mean) = mean_count {
                    let variance = Sample::calculate_uncorrected_variance_usize(&word_counts, mean)
                        .expect("Word count should have a variance when there are solutions.");
                    let std_dev = variance.sqrt();
                    (Ok(variance), Ok(std_dev))
                } else {
                    (
                        Err(StatsError::new(
                            String::from("could not calculate count variance, mean not available"),
                            ErrorKind::MissingValue,
                            None,
                        )),
                        Err(StatsError::new(
                            String::from(
                                "could not calculate count standard deviation, mean not available",
                            ),
                            ErrorKind::MissingValue,
                            None,
                        )),
                    )
                };

            let mean_duration = Sample::calculate_mean_duration(&durations)
                .expect("Durations should have a mean when there are solutions.");
            let (uncorrected_duration_variance, uncorrected_duration_standard_deviation) =
                if let Ok(mean) = mean_duration {
                    let variance =
                        Sample::calculate_uncorrected_variance_duration(&durations, &mean)
                            .expect("Durations should have a variance when there are solutions.");
                    let std_dev = variance
                    .as_ref()
                    .map(|variance| Sample::nanos_squared_to_secs_squared(*variance).sqrt())
                    .map_err(|_e| {
                    StatsError::new(
                        String::from(
                            "could not calculate Duration standard deviation, variance not available",
                        ),
                        ErrorKind::MissingValue,
                        None,
                    )});
                    (variance, std_dev)
                } else {
                    (
			Err(StatsError::new(
			    String::from("could not calculate Duration variance, mean not available"),
			    ErrorKind::MissingValue,
			    None
			)),
			Err(StatsError::new(
			    String::from("could not calculate Duration standard deviation, mean not available"),
			    ErrorKind::MissingValue,
			    None
			)),
		    )
                };

            Some(SolutionStats {
                mean_length,
                median_length: Sample::calculate_median_usize(&mut lengths)
                    .expect("Lengths should have a median when there are solutions."),
                min_length: lengths[0],
                max_length: lengths[successes - 1],
                uncorrected_length_variance,
                uncorrected_length_standard_deviation,
                mean_count,
                median_count: Sample::calculate_median_usize(&mut word_counts)
                    .expect("Word count should have a median when there are solutions."),
                min_count: word_counts[0],
                max_count: word_counts[successes - 1],
                uncorrected_count_variance,
                uncorrected_count_standard_deviation,
                mean_duration,
                median_duration: Sample::calculate_median_duration(&mut durations)
                    .expect("Durations should have a median when there are solutions."),
                min_duration: durations[0],
                max_duration: durations[successes - 1],
                uncorrected_duration_variance,
                uncorrected_duration_standard_deviation,
            })
        } else {
            None
        };

        Stats {
            successes,
            failures,
            solution_stats,
        }
    }

    /// Validates all solutions in the sample and returns the number of solutions that failed.
    ///
    /// Note that a `Timing` that resulted in an error does not count as an invalid solution.
    pub fn validate_solutions(&self, model: &Model) -> usize {
        let mut failures = 0;

        for timing in self.timings.iter() {
            if let Ok(solution) = timing.result.as_ref() {
                if !solution.is_valid(model) {
                    failures += 1;
                };
            };
        }

        failures
    }

    /// Find the shortest Solution in the Sample.
    ///
    /// Returns None if there are no Solutions.
    pub fn best_solution(&self) -> Option<(&Solution, &Duration)> {
        self.timings
            .iter()
            .filter_map(|x| {
                if let Ok(s) = &x.result {
                    Some((s, &x.duration))
                } else {
                    None
                }
            })
            .min_by(|x, y| x.0.len().cmp(&y.0.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod timing {
        use super::*;
        use crate::{solution::create_test_solution, solvers::ErrorKind as SolverErrorKind};

        fn initialize() -> (Model, Timing, Timing) {
            let model = Model::build_test_model("cat abs cab");
            let duration = Duration::new(5, 345);
            return (
                model.clone(),
                Timing {
                    result: Ok(create_test_solution(&model, vec!["cat", "abs"])),
                    duration: duration.clone(),
                },
                Timing {
                    result: Err(SolverError::new(
                        String::from("test"),
                        SolverErrorKind::ModelConsistencyError,
                        None,
                    )),
                    duration,
                },
            );
        }

        #[test]
        fn partial_eq_equal_solution() {
            let init = initialize();
            assert_eq!(
                init.1,
                Timing {
                    result: Ok(init.1.result.as_ref().unwrap().clone()),
                    duration: init.1.duration.clone()
                }
            )
        }

        #[test]
        fn partial_eq_equal_errors() {
            let init = initialize();
            let duration = Duration::new(5, 345);
            let t2 = Timing {
                result: Err(SolverError::new(
                    String::from("test"),
                    SolverErrorKind::ModelConsistencyError,
                    None,
                )),
                duration,
            };
            assert_eq!(init.2, t2);
        }

        #[test]
        fn partial_eq_unequal_duration() {
            let init = initialize();
            assert_ne!(
                init.1,
                Timing {
                    result: Ok(init.1.result.as_ref().unwrap().clone()),
                    duration: Duration::new(99, 999)
                }
            );
        }

        #[test]
        fn partial_eq_unequal_solution() {
            let init = initialize();
            assert_ne!(
                init.1,
                Timing {
                    result: Ok(create_test_solution(&init.0, vec!["cat"])),
                    duration: init.1.duration.clone()
                }
            );
        }

        #[test]
        fn partial_eq_unequal_result_types() {
            let init = initialize();
            assert_ne!(init.1, init.2);
        }
    }

    mod stats {
        use super::*;

        fn build_test_solution_stats() -> Option<SolutionStats> {
            Some(SolutionStats {
                mean_length: Ok(1.0),
                median_length: 2.0,
                min_length: 3,
                max_length: 4,
                uncorrected_length_variance: Ok(21.0),
                uncorrected_length_standard_deviation: Ok(22.0),
                mean_count: Ok(5.0),
                median_count: 6.0,
                min_count: 7,
                max_count: 8,
                uncorrected_count_variance: Ok(23.0),
                uncorrected_count_standard_deviation: Ok(24.0),
                mean_duration: Ok(Duration::new(9, 0)),
                median_duration: Duration::new(10, 0),
                min_duration: Duration::new(11, 0),
                max_duration: Duration::new(12, 0),
                uncorrected_duration_variance: Ok(25),
                uncorrected_duration_standard_deviation: Ok(26.0),
            })
        }

        fn build_test_error() -> StatsError {
            StatsError::new(String::from("test"), ErrorKind::MissingValue, None)
        }

        fn build_error_solution_stats() -> Option<SolutionStats> {
            Some(SolutionStats {
                mean_length: Err(build_test_error()),
                median_length: 2.0,
                min_length: 3,
                max_length: 4,
                uncorrected_length_variance: Err(build_test_error()),
                uncorrected_length_standard_deviation: Err(build_test_error()),
                mean_count: Err(build_test_error()),
                median_count: 6.0,
                min_count: 7,
                max_count: 8,
                uncorrected_count_variance: Err(build_test_error()),
                uncorrected_count_standard_deviation: Err(build_test_error()),
                mean_duration: Err(build_test_error()),
                median_duration: Duration::new(10, 0),
                min_duration: Duration::new(11, 0),
                max_duration: Duration::new(12, 0),
                uncorrected_duration_variance: Err(build_test_error()),
                uncorrected_duration_standard_deviation: Err(build_test_error()),
            })
        }

        fn build_stats_failures() -> Stats {
            Stats {
                successes: 32,
                failures: 0,
                solution_stats: None,
            }
        }

        fn build_stats_successes() -> Stats {
            Stats {
                successes: 32,
                failures: 0,
                solution_stats: build_test_solution_stats(),
            }
        }

        fn build_stats_errors() -> Stats {
            Stats {
                successes: 32,
                failures: 0,
                solution_stats: build_error_solution_stats(),
            }
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
            let stats = build_stats_failures();
            assert!(stats.mean_length().is_none());
        }

        #[test]
        fn mean_length_some() {
            let stats = build_stats_successes();
            assert_eq!(
                *stats
                    .mean_length()
                    .expect("Test stats should have solution stat block."),
                Ok(1.0)
            );
        }

        #[test]
        fn mean_length_error() {
            let stats = build_stats_errors();
            assert_eq!(stats.mean_length(), Some(&Err(build_test_error())));
        }

        #[test]
        fn median_length_none() {
            let stats = build_stats_failures();
            assert!(stats.median_length().is_none());
        }

        #[test]
        fn median_length_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .median_length()
                    .expect("Test stats should have solution stat block."),
                2.0
            );
        }

        #[test]
        fn min_length_none() {
            let stats = build_stats_failures();
            assert!(stats.min_length().is_none());
        }

        #[test]
        fn min_length_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .min_length()
                    .expect("Test stats should have solution stat block."),
                3
            );
        }

        #[test]
        fn max_length_none() {
            let stats = build_stats_failures();
            assert!(stats.max_length().is_none());
        }

        #[test]
        fn max_length_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .max_length()
                    .expect("Test stats should have solution stat block."),
                4
            );
        }

        #[test]
        fn uncorrected_length_variance_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_length_variance().is_none());
        }

        #[test]
        fn uncorrected_length_variance_some() {
            let stats = build_stats_successes();
            assert_eq!(stats.uncorrected_length_variance(), Some(&Ok(21.0)));
        }

        #[test]
        fn uncorrected_length_variance_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_length_variance(),
                Some(&Err(build_test_error()))
            );
        }

        #[test]
        fn uncorrected_length_standard_deviation_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_length_standard_deviation().is_none());
        }

        #[test]
        fn uncorrected_length_standard_deviation_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats.uncorrected_length_standard_deviation(),
                Some(&Ok(22.0))
            );
        }

        #[test]
        fn uncorrected_length_standard_deviation_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_length_standard_deviation(),
                Some(&Err(build_test_error()))
            );
        }

        #[test]
        fn mean_count_none() {
            let stats = build_stats_failures();
            assert!(stats.mean_count().is_none());
        }

        #[test]
        fn mean_count_some() {
            let stats = build_stats_successes();
            assert_eq!(stats.mean_count(), Some(&Ok(5.0)));
        }

        #[test]
        fn mean_count_error() {
            let stats = build_stats_errors();
            assert_eq!(stats.mean_count(), Some(&Err(build_test_error())));
        }

        #[test]
        fn median_count_none() {
            let stats = build_stats_failures();
            assert!(stats.median_count().is_none());
        }

        #[test]
        fn median_count_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .median_count()
                    .expect("Test stats should have solution stat block."),
                6.0
            );
        }

        #[test]
        fn min_count_none() {
            let stats = build_stats_failures();
            assert!(stats.min_count().is_none());
        }

        #[test]
        fn min_count_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .min_count()
                    .expect("Test stats should have solution stat block."),
                7
            );
        }

        #[test]
        fn max_count_none() {
            let stats = build_stats_failures();
            assert!(stats.max_count().is_none());
        }

        #[test]
        fn max_count_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .max_count()
                    .expect("Test stats should have solution stat block."),
                8
            );
        }

        #[test]
        fn uncorrected_count_variance_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_count_variance().is_none());
        }

        #[test]
        fn uncorrected_count_variance_some() {
            let stats = build_stats_successes();
            assert_eq!(stats.uncorrected_count_variance(), Some(&Ok(23.0)));
        }

        #[test]
        fn uncorrected_count_variance_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_count_variance(),
                Some(&Err(build_test_error()))
            );
        }

        #[test]
        fn uncorrected_count_standard_deviation_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_count_standard_deviation().is_none());
        }

        #[test]
        fn uncorrected_count_standard_deviation_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats.uncorrected_count_standard_deviation(),
                Some(&Ok(24.0))
            );
        }

        #[test]
        fn uncorrected_count_standard_deviation_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_count_standard_deviation(),
                Some(&Err(build_test_error()))
            );
        }

        #[test]
        fn mean_duration_none() {
            let stats = build_stats_failures();
            assert!(stats.mean_duration().is_none());
        }

        #[test]
        fn mean_duration_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .mean_duration()
                    .expect("Test stats should have solution stat block."),
                &Ok(Duration::new(9, 0)),
            );
        }

        #[test]
        fn mean_duration_error() {
            let stats = build_stats_errors();
            assert_eq!(stats.mean_duration(), Some(&Err(build_test_error())));
        }

        #[test]
        fn median_duration_none() {
            let stats = build_stats_failures();
            assert!(stats.median_duration().is_none());
        }

        #[test]
        fn median_duration_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .median_duration()
                    .expect("Test stats should have solution stat block."),
                Duration::new(10, 0),
            );
        }

        #[test]
        fn min_duration_none() {
            let stats = build_stats_failures();
            assert!(stats.min_duration().is_none());
        }

        #[test]
        fn min_duration_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .min_duration()
                    .expect("Test stats should have solution stat block."),
                Duration::new(11, 0),
            );
        }

        #[test]
        fn max_duration_none() {
            let stats = build_stats_failures();
            assert!(stats.max_duration().is_none());
        }

        #[test]
        fn max_duration_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats
                    .max_duration()
                    .expect("Test stats should have solution stat block."),
                Duration::new(12, 0),
            );
        }

        #[test]
        fn uncorrected_duration_variance_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_duration_variance().is_none());
        }

        #[test]
        fn uncorrected_duration_variance_some() {
            let stats = build_stats_successes();
            assert_eq!(stats.uncorrected_duration_variance(), Some(&Ok(25)));
        }

        #[test]
        fn uncorrected_duration_variance_some_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_duration_variance(),
                Some(&Err(build_test_error()))
            );
        }

        #[test]
        fn uncorrected_duration_standard_deviation_none() {
            let stats = build_stats_failures();
            assert!(stats.uncorrected_duration_standard_deviation().is_none());
        }

        #[test]
        fn uncorrected_duration_standard_deviation_some() {
            let stats = build_stats_successes();
            assert_eq!(
                stats.uncorrected_duration_standard_deviation(),
                Some(&Ok(26.0))
            );
        }

        #[test]
        fn uncorrected_duration_standard_deviation_error() {
            let stats = build_stats_errors();
            assert_eq!(
                stats.uncorrected_duration_standard_deviation(),
                Some(&Err(build_test_error()))
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
            let expected = "Solution generation statistics:\nTotal Runs: 32\nSuccesses: 24\nFailures: 8\nmean length: 1\nmedian_length: 2\nmin_length: 3\nmax_length: 4\nuncorrected standard deviation: 22\nmean count: 5\nmedian_count: 6\nmin_count: 7\nmax_count: 8\nuncorrected standard deviation: 24\nmean compute time: 9s\nmedian compute time: 10s\nmin compute time: 11s\nmax compute time: 12s\nuncorrected standard deviation: 26s";
            assert_eq!(stats.to_string(), expected)
        }
    }

    mod sample_tests {
        use crate::{
            consts::BILLION, solution::create_test_solution, solvers,
            solvers::ErrorKind as SolverErrorKind,
        };
        use std::io::BufReader;

        use super::*;

        fn error_solver(_model: &Model) -> Result<Solution, SolverError> {
            Err(SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            ))
        }

        #[test]
        fn calculate_stats() {
            let model = Model::build_test_model("cat abs cab");
            let s1 = create_test_solution(&model, vec!["cat", "abs", "cab"]);
            let t1_duration = Duration::new(1, 0);
            let t1 = Timing {
                result: Ok(s1.clone()),
                duration: t1_duration,
            };

            let s2 = create_test_solution(&model, vec!["cat", "abs"]);
            let t2_duration = Duration::new(2, 0);
            let t2 = Timing {
                result: Ok(s2.clone()),
                duration: t2_duration,
            };

            let s3 = SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            );
            let t3_duration = Duration::new(10000, 0);
            let t3 = Timing {
                result: Err(s3),
                duration: t3_duration,
            };

            let s4 = create_test_solution(&model, vec!["cat", "abs", "cab"]);
            let t4_duration = Duration::new(3, 0);
            let t4 = Timing {
                result: Ok(s4.clone()),
                duration: t4_duration,
            };
            let sample = Sample {
                timings: vec![t1, t2, t3, t4],
            };
            let stats = sample.calculate_stats();

            let expected_mean_length = (s1.len() + s2.len() + s4.len()) as f64 / 3.0;
            let expected_length_variance = ((s1.len() as f64 - expected_mean_length).powi(2)
                + (s2.len() as f64 - expected_mean_length).powi(2)
                + (s4.len() as f64 - expected_mean_length).powi(2))
                / 3.0;

            let expected_mean_count =
                (s1.words().len() + s2.words().len() + s4.words().len()) as f64 / 3.0;
            let expected_count_variance = ((s1.words().len() as f64 - expected_mean_count).powi(2)
                + (s2.words().len() as f64 - expected_mean_count).powi(2)
                + (s4.words().len() as f64 - expected_mean_count).powi(2))
                / 3.0;

            let expected_mean_duration = (t1_duration + t2_duration + t4_duration) / 3;
            let emd_nanos = expected_mean_duration.as_nanos();
            let expected_duration_variance = ((emd_nanos - t1_duration.as_nanos()).pow(2)
                + (t2_duration.as_nanos() - emd_nanos).pow(2)
                + (t4_duration.as_nanos() - emd_nanos).pow(2))
                / 3;

            let expected_solution_stats = SolutionStats {
                mean_length: Ok(expected_mean_length),
                median_length: s4.len() as f64,
                min_length: s2.len(),
                max_length: s4.len(),
                uncorrected_length_variance: Ok(expected_length_variance),
                uncorrected_length_standard_deviation: Ok(expected_length_variance.sqrt()),
                mean_count: Ok(expected_mean_count),
                median_count: s4.words().len() as f64,
                min_count: s2.words().len(),
                max_count: s4.words().len(),
                uncorrected_count_variance: Ok(expected_count_variance),
                uncorrected_count_standard_deviation: Ok(expected_count_variance.sqrt()),
                mean_duration: Ok(expected_mean_duration),
                median_duration: t2_duration,
                min_duration: t1_duration,
                max_duration: t4_duration,
                uncorrected_duration_variance: Ok(expected_duration_variance),
                uncorrected_duration_standard_deviation: Ok(Sample::nanos_squared_to_secs_squared(
                    expected_duration_variance,
                )
                .sqrt()),
            };

            let expected_stats = Stats {
                successes: 3,
                failures: 1,
                solution_stats: Some(expected_solution_stats),
            };
            assert_eq!(stats.runs(), 4);
            assert_eq!(stats, expected_stats);
        }

        #[test]
        fn calculate_stats_all_errors() {
            let s1 = SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            );
            let mut duration = Duration::new(20000, 0);
            let t1 = Timing {
                result: Err(s1),
                duration,
            };

            let s2 = SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            );
            duration = Duration::new(10000, 0);
            let t2 = Timing {
                result: Err(s2),
                duration,
            };
            let sample = Sample {
                timings: vec![t1, t2],
            };
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), 2);
            assert_eq!(stats.successes(), 0);
            assert_eq!(stats.failures(), 2);
            assert_eq!(stats.mean_length(), None);
            assert_eq!(stats.median_length(), None);
            assert_eq!(stats.min_length(), None);
            assert_eq!(stats.max_length(), None);
            assert_eq!(stats.mean_count(), None);
            assert_eq!(stats.median_count(), None);
            assert_eq!(stats.min_count(), None);
            assert_eq!(stats.max_count(), None);
            assert_eq!(stats.mean_duration(), None);
            assert_eq!(stats.median_duration(), None);
            assert_eq!(stats.min_duration(), None);
            assert_eq!(stats.max_duration(), None);
        }

        #[test]
        fn empty() {
            let runs = 30;
            let input = BufReader::new(std::io::empty());
            let model = Model::new(input).expect("Model creation should not fail");
            let sample = Sample::new(solvers::whole_text, &model, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.successes(), runs);
            assert_eq!(stats.max_count(), Some(0));
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn one_run() {
            let runs = 1;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.successes(), runs);
            assert_eq!(stats.max_count(), Some(3));
            assert_eq!(stats.min_count(), Some(3));
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn one_run_extra_words() {
            let runs = 1;
            let m1 = Model::build_test_model("cat cabs");
            let m2 = Model::build_test_model("cat abs cabs");
            let sample = Sample::new(solvers::whole_text, &m1, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.successes(), runs);
            assert_eq!(stats.max_count(), Some(2));
            assert_eq!(stats.min_count(), Some(2));
            assert_eq!(sample.validate_solutions(&m2), 0);
        }

        #[test]
        fn multiple_runs() {
            let runs = 30;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.successes(), runs);
            assert_eq!(stats.max_count(), Some(3));
            assert_eq!(stats.min_count(), Some(3));
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn error() {
            let runs = 1;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(error_solver, &model, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.failures(), runs);
        }

        #[test]
        fn error_multiple() {
            let runs = 30;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(error_solver, &model, runs);
            let stats = sample.calculate_stats();
            assert_eq!(stats.runs(), runs);
            assert_eq!(stats.failures(), runs);
        }

        #[test]
        fn validate_empty() {
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, 0);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_one_success() {
            let runs = 1;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, runs);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_one_error() {
            let runs = 1;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(error_solver, &model, runs);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_multiple_successes() {
            let runs = 30;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, runs);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_multiple_errors() {
            let runs = 30;
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(error_solver, &model, runs);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_mix_error() {
            let runs = 30;
            let model = Model::build_test_model("cat abs cab");
            let mut sample = Sample::new(solvers::whole_text, &model, runs);
            let start = Instant::now();
            let error = SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            );
            let duration = start.elapsed();
            let timing = Timing {
                result: Err(error),
                duration,
            };
            sample.timings.push(timing);
            assert_eq!(sample.validate_solutions(&model), 0);
        }

        #[test]
        fn validate_one_invalid() {
            let runs = 1;
            let m1 = Model::build_test_model("cat");
            let m2 = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &m1, runs);
            assert_eq!(sample.validate_solutions(&m2), runs);
        }

        #[test]
        fn validate_multiple_invalid() {
            let runs = 30;
            let m1 = Model::build_test_model("cat");
            let m2 = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &m1, runs);
            assert_eq!(sample.validate_solutions(&m2), runs);
        }

        #[test]
        fn validate_mix_invalid() {
            let runs = 30;
            let m1 = Model::build_test_model("cat");
            let m2 = Model::build_test_model("cat abs cab");
            let mut sample = Sample::new(solvers::whole_text, &m2, runs);

            let mut s2 = Sample::new(solvers::whole_text, &m1, 1);
            sample
                .timings
                .push(s2.timings.pop().expect("Sample creation should succeed."));
            let start = Instant::now();
            let error = SolverError::new(
                String::from("test"),
                SolverErrorKind::ModelConsistencyError,
                None,
            );

            let duration = start.elapsed();
            let timing = Timing {
                result: Err(error),
                duration,
            };
            sample.timings.push(timing);

            assert_eq!(sample.validate_solutions(&m2), 1);
        }

        #[test]
        fn best_solution_empty() {
            let model = Model::build_test_model("cat abs cab");
            let sample = Sample::new(solvers::whole_text, &model, 0);
            assert!(sample.best_solution().is_none());
        }

        #[test]
        fn best_solution_one() {
            let model = Model::build_test_model("cat abs cab");
            let start = Instant::now();
            let solution = create_test_solution(&model, vec!["cat", "abs"]);
            let duration = start.elapsed();
            let best_duration = duration.clone();
            let timing = Timing {
                result: Ok(solution.clone()),
                duration,
            };
            let sample = Sample {
                timings: vec![timing],
            };
            let best = sample
                .best_solution()
                .expect("Should find the test solution.");
            assert_eq!(best, (&solution, &best_duration));
        }

        #[test]
        fn best_solution_multiple() {
            let model = Model::build_test_model("cat abs cab");
            let mut start = Instant::now();
            let s1 = create_test_solution(&model, vec!["cat", "abs", "cab"]);
            let mut duration = start.elapsed();
            let t1 = Timing {
                result: Ok(s1.clone()),
                duration,
            };
            start = Instant::now();
            let s2 = create_test_solution(&model, vec!["cat", "abs"]);
            duration = start.elapsed();
            let best_duration = duration.clone();
            let t2 = Timing {
                result: Ok(s2.clone()),
                duration,
            };
            start = Instant::now();
            let s3 = create_test_solution(&model, vec!["cat", "abs", "cab"]);
            duration = start.elapsed();
            let t3 = Timing {
                result: Ok(s3.clone()),
                duration,
            };
            let sample = Sample {
                timings: vec![t1, t2, t3],
            };
            let best = sample
                .best_solution()
                .expect("Should find the test solution.");
            assert_eq!(best, (&s2, &best_duration));
        }

        #[test]
        fn calculate_mean_usize_empty() {
            let nums = Vec::new();
            assert_eq!(Sample::calculate_mean_usize(&nums), None);
        }

        #[test]
        fn calculate_mean_usize_one() {
            let nums = vec![3];
            assert_eq!(Sample::calculate_mean_usize(&nums), Some(Ok(3.0)));
        }

        #[test]
        fn calculate_mean_usize_multiple() {
            let nums = vec![1, 3, 5, 7];
            assert_eq!(Sample::calculate_mean_usize(&nums), Some(Ok(4.0)));
        }

        #[test]
        fn calculate_mean_usize_overflow() {
            let nums = vec![usize::MAX, 1];
            let result = Sample::calculate_mean_usize(&nums)
                .expect("should produce Some for non-empty input");
            assert!(result.is_err(), "{:?} is not Err", result);
        }

        #[test]
        fn calculate_median_usize_empty() {
            let mut nums = Vec::new();
            assert_eq!(Sample::calculate_median_usize(&mut nums), None);
        }

        #[test]
        fn calculate_median_usize_one() {
            let mut nums = vec![3];
            assert_eq!(Sample::calculate_median_usize(&mut nums), Some(3.0));
        }

        #[test]
        fn calculate_median_usize_odd() {
            let mut nums = vec![7, 3, 5, 4, 1, 6, 2];
            assert_eq!(Sample::calculate_median_usize(&mut nums), Some(4.0));
        }

        #[test]
        fn calculate_median_usize_even() {
            let mut nums = vec![11, 8, 4, 12];
            assert_eq!(Sample::calculate_median_usize(&mut nums), Some(9.5));
        }

        #[test]
        fn calculate_uncorrected_variance_usize_empty() {
            let nums = Vec::new();
            assert_eq!(
                Sample::calculate_uncorrected_variance_usize(&nums, 0.0),
                None
            );
        }

        #[test]
        fn calculate_uncorrected_variance_usize_one() {
            let nums = vec![7];
            assert_eq!(
                Sample::calculate_uncorrected_variance_usize(&nums, 7.0),
                Some(0.0)
            );
        }

        #[test]
        fn calculate_uncorrected_variance_usize_multiple() {
            let nums = vec![138, 4, 7, 54];
            let mean = Sample::calculate_mean_usize(&nums)
                .expect("non-empty input should produce output")
                .expect("should not overflow");
            let expected_mean = (138 + 4 + 7 + 54) as f64 / 4.0;
            let variance = ((138.0 - expected_mean).powi(2)
                + (4.0 - expected_mean).powi(2)
                + (7.0 - expected_mean).powi(2)
                + (54.0 - expected_mean).powi(2))
                / 4.0;
            assert_eq!(
                Sample::calculate_uncorrected_variance_usize(&nums, mean),
                Some(variance)
            );
        }

        #[test]
        fn calculate_mean_duration_empty() {
            let durations = Vec::new();
            assert_eq!(Sample::calculate_mean_duration(&durations), None);
        }

        #[test]
        fn calculate_mean_duration_one() {
            let durations = vec![Duration::new(3, 2507)];
            assert_eq!(
                Sample::calculate_mean_duration(&durations),
                Some(Ok(Duration::new(3, 2507)))
            );
        }

        #[test]
        fn calculate_mean_duration_multiple() {
            let durations = vec![
                Duration::new(1, 994),
                Duration::new(3, 1),
                Duration::new(5, 2),
                Duration::new(7, 3),
            ];
            assert_eq!(
                Sample::calculate_mean_duration(&durations),
                Some(Ok(Duration::new(4, 250)))
            );
        }

        #[test]
        fn calculate_mean_duration_overflow() {
            let durations = vec![Duration::MAX; 4];
            let mean = Sample::calculate_mean_duration(&durations)
                .expect("Calculate mean should return as result on non-empty input.");
            assert!(mean.is_err());
            assert_eq!(
                *mean.unwrap_err().kind(),
                ErrorKind::OverflowError(OverflowLocation::MeanSummation)
            );
        }

        #[test]
        fn calculate_median_duration_empty() {
            let mut durations = Vec::new();
            assert_eq!(Sample::calculate_median_duration(&mut durations), None);
        }

        #[test]
        fn calculate_median_duration_one() {
            let mut durations = vec![Duration::new(3, 7)];
            assert_eq!(
                Sample::calculate_median_duration(&mut durations),
                Some(Duration::new(3, 7))
            );
        }

        #[test]
        fn calculate_median_duration_odd() {
            let mut durations = vec![
                Duration::new(3, 289),
                Duration::new(2, 157893),
                Duration::new(4, 567),
                Duration::new(1, 8934),
                Duration::new(5, 16348),
                Duration::new(7, 246),
                Duration::new(6, 234),
            ];
            assert_eq!(
                Sample::calculate_median_duration(&mut durations),
                Some(Duration::new(4, 567))
            );
        }

        #[test]
        fn calculate_median_duration_even() {
            let mut durations = vec![
                Duration::new(8, 50),
                Duration::new(4, 786),
                Duration::new(11, 200),
                Duration::new(12, 760),
            ];
            assert_eq!(
                Sample::calculate_median_duration(&mut durations),
                Some(Duration::new(9, 500000125))
            );
        }

        #[test]
        fn calculate_uncorrected_variance_duration_empty() {
            let mut durations = Vec::new();
            let mean = Duration::new(0, 0);
            assert_eq!(
                Sample::calculate_uncorrected_variance_duration(&mut durations, &mean),
                None
            );
        }

        #[test]
        fn calculate_uncorrected_variance_duration_one() {
            let mean = Duration::new(10, 50);
            let mut durations = vec![mean.clone()];
            assert_eq!(
                Sample::calculate_uncorrected_variance_duration(&mut durations, &mean),
                Some(Ok(0)),
            );
        }

        #[test]
        fn calculate_uncorrected_variance_duration_multiple() {
            let bill = BILLION as u64;
            let one = 10 * bill + 50 as u64;
            let two = 20 * bill + 123 as u64;
            let three = 100 * bill + 85213 as u64;
            let four = 24 * bill + 654 as u64;

            let mut durations = vec![
                Duration::from_nanos(one),
                Duration::from_nanos(two),
                Duration::from_nanos(three),
                Duration::from_nanos(four),
            ];

            let mean = (one + two + three + four) / 4;
            let mean_duration = Sample::calculate_mean_duration(&durations)
                .expect("Non empty sample should have mean.")
                .expect("mean should not overflow");
            assert_eq!(
                Sample::calculate_uncorrected_variance_duration(&mut durations, &mean_duration),
                Some(Ok((((one as i128 - mean as i128).pow(2)
                    + (two as i128 - mean as i128).pow(2)
                    + (three as i128 - mean as i128).pow(2)
                    + (four as i128 - mean as i128).pow(2))
                    / 4) as u128)),
            );
        }

        #[test]
        fn calculate_uncorrected_variance_duration_multiple_fractional() {
            let bill = BILLION as f64;
            let one = (0.3455559 * bill) as u64;
            let two = (0.3469258 * bill) as u64;
            let three = (0.3461224 * bill) as u64;
            let four = (0.34611992 * bill) as u64;

            let mut durations = vec![
                Duration::from_nanos(one),
                Duration::from_nanos(two),
                Duration::from_nanos(three),
                Duration::from_nanos(four),
            ];

            let mean = ((one + two + three + four) / 4) as i128;
            let mean_duration = Sample::calculate_mean_duration(&durations)
                .expect("Non empty sample should have mean.")
                .expect("mean should not overflow");
            assert_eq!(
                Sample::calculate_uncorrected_variance_duration(&mut durations, &mean_duration),
                Some(Ok((((one as i128 - mean).pow(2)
                    + (two as i128 - mean).pow(2)
                    + (three as i128 - mean).pow(2)
                    + (four as i128 - mean).pow(2))
                    / 4) as u128)),
            );
        }

        #[test]
        fn calculate_uncorrected_variance_overflow_power() {
            let mut durations = vec![Duration::ZERO; 10000];
            durations.push(Duration::MAX);
            let mean = Sample::calculate_mean_duration(&durations)
                .expect("Non empty sample should have mean.")
                .expect("mean should not overflow");
            let variance = Sample::calculate_uncorrected_variance_duration(&durations, &mean)
                .expect("Calculation should produce result for non empty input.");
            assert!(variance.is_err());
            assert_eq!(
                *variance.unwrap_err().kind(),
                ErrorKind::OverflowError(OverflowLocation::DeviationSquaring)
            );
        }

        #[test]
        fn calculate_uncorrected_variance_overflow_sum() {
            let duration = Duration::from_nanos(u64::MAX);
            let mut durations = vec![duration; 100];
            let mut zeros = vec![Duration::ZERO; 100];
            durations.append(&mut zeros);
            let mean = Duration::from_nanos((duration.as_nanos() / 2) as u64);
            let variance = Sample::calculate_uncorrected_variance_duration(&durations, &mean)
                .expect("Calculation should produce result for non empty input.");
            assert!(variance.is_err());
            assert_eq!(
                *variance.unwrap_err().kind(),
                ErrorKind::OverflowError(OverflowLocation::SquareDeviationSummation)
            );
        }

        #[test]
        fn nanos_squared_to_secs_squared_zero() {
            assert_eq!(Sample::nanos_squared_to_secs_squared(0), 0 as f64);
        }

        #[test]
        fn nanos_squared_to_secs_squared_zero_secs() {
            assert_eq!(
                Sample::nanos_squared_to_secs_squared(401_781_847),
                0.000000000401781847
            );
        }

        #[test]
        fn nanos_squared_to_secs_squared_zero_nanos() {
            assert_eq!(
                Sample::nanos_squared_to_secs_squared(10_000_000_000),
                0.00000001
            );
        }

        #[test]
        fn nanos_squared_to_secs_squared_both() {
            assert_eq!(
                Sample::nanos_squared_to_secs_squared(6_000_345_738),
                0.000000006000345738
            );
        }
    }
}
