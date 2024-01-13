//! Algorithms that solve the character pair problem.

use crate::{
    error::GenericError,
    solution::Solution,
    text_model::{CharacterPair, Model, Word},
};
use std::{collections::HashSet, rc::Rc};

/// An error returned by a solver.
pub type SolverError = GenericError<ErrorKind>;

/// An enum specifying the category of error.
#[derive(Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// The text Model is inconsistent.
    ///
    /// e.g. the solution is not complete, but there are no more unchosen words to choose from.
    ModelConsistencyError,
}

/// Creates a `Solution` from a `Model` by choosing all words in the `Model`.
pub fn whole_text(model: &Model) -> Result<Solution, SolverError> {
    Ok(Solution::new(model.words().clone()))
}

fn word_score_new_pairs(word: &Word, unchosen_pairs: &HashSet<Rc<CharacterPair>>) -> usize {
    let mut duplicates = HashSet::new();
    let mut score = 0;
    for pair in word.pairs() {
        if unchosen_pairs.contains(pair) && !duplicates.contains(pair) {
            score += 1;
            duplicates.insert(pair);
        }
    }
    score
}

// This should be tested with every new scoring algorithm.
fn find_best_word(
    judge: &dyn Fn(&Word, &HashSet<Rc<CharacterPair>>) -> usize,
    unchosen_words: &HashSet<Rc<Word>>,
    unchosen_pairs: &HashSet<Rc<CharacterPair>>,
) -> Option<Rc<Word>> {
    let mut words_iter = unchosen_words.iter();
    if let Some(mut best_word) = words_iter.next() {
        let mut high_score = judge(best_word, unchosen_pairs);

        for word in words_iter {
            let score = judge(word, unchosen_pairs);
            if (score > high_score)
                || (score == high_score
                    && (word.len() < best_word.len()
                        || (word.len() == best_word.len() && word < best_word)))
            {
                high_score = score;
                best_word = word;
            };
        }

        return Some(best_word.clone());
    };
    None
}

/// Creates a `Solution` from a `text_model::Model` by repeatedly choosing
/// the most valuable `Word`.
///
/// The value of a word is determined by how many unchosen `CharacterPairs`
/// it contains.
pub fn greedy_most_valuable_word(model: &Model) -> Result<Solution, SolverError> {
    let unique_words = match model.find_words_with_unique_pairs() {
        Ok(words) => words,
        Err(err) => {
            return Err(SolverError::new(
                String::from("could not compute solution, model is internally inconsistent"),
                ErrorKind::ModelConsistencyError,
                Some(Box::new(err)),
            ))
        }
    };

    let mut solution = Solution::new(unique_words);
    let mut unchosen_pairs = solution.unchosen_pairs(model);
    let mut unchosen_words = solution.unchosen_words(model);

    while !unchosen_pairs.is_empty() {
        if let Some(best_word) =
            find_best_word(&word_score_new_pairs, &unchosen_words, &unchosen_pairs)
        {
            unchosen_words.remove(&best_word);
            for pair in best_word.pairs() {
                unchosen_pairs.remove(pair);
            }
            solution.add_word(best_word);
        } else {
            return Err(SolverError::new(
		format!("Model is inconsistant, there are {} pairs remaining to cover, but no more unchosen words.", unchosen_pairs.len()),
		ErrorKind::ModelConsistencyError,
		None,
            ));
        }
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    mod whole_text {
        use super::*;

        #[test]
        fn empty() {
            let model = Model::new(io::empty()).expect("reading empty should not fail");
            let solution = whole_text(&model).expect("test should produce solution");
            assert_eq!(solution.words(), model.words());
        }

        #[test]
        fn words() {
            let model = Model::build_test_model("cat abs hutch oven cab cap much coven");
            let solution = whole_text(&model).expect("test should produce solution");
            assert_eq!(solution.words(), model.words());
        }
    }

    mod greedy_algorithms {
        use super::*;
        use crate::solution::build_test_model_and_solution;

        fn build_test_fixtures(
            model_text: &str,
            chosen_words: Vec<&str>,
            word: &str,
        ) -> (Model, Solution, Rc<Word>) {
            let (model, solution) = build_test_model_and_solution(model_text, chosen_words);
            let word = model
                .find_word_str(word)
                .expect(&format!("should contain \"{}\"", word));
            (model, solution, word)
        }

        #[test]
        fn find_best_word_empty_all() {
            let (model, solution) =
                build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
            let best_word = find_best_word(
                &word_score_new_pairs,
                &solution.unchosen_words(&model),
                &solution.unchosen_pairs(&model),
            );
            assert_eq!(best_word, None);
        }

        #[test]
        fn find_best_word_empty_words() {
            let (model, solution) =
                build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
            let best_word = find_best_word(
                &word_score_new_pairs,
                &solution.unchosen_words(&model),
                &solution.pairs(),
            );
            assert_eq!(best_word, None);
        }

        #[test]
        fn find_best_word_empty_pairs() {
            let (model, solution, word) = build_test_fixtures("cat abs cab", Vec::new(), "abs");
            let best_word = find_best_word(
                &word_score_new_pairs,
                &solution.unchosen_words(&model),
                &HashSet::new(),
            );
            assert_eq!(best_word, Some(word));
        }

        mod greedy_most_valuable_word {
            use super::*;

            #[test]
            fn word_score_new_pairs_all_chosen() {
                let (model, solution, word) =
                    build_test_fixtures("cat at ca", vec!["at", "ca"], "cat");
                assert_eq!(
                    word_score_new_pairs(&word, &solution.unchosen_pairs(&model)),
                    0
                );
            }

            #[test]
            fn word_score_new_pairs_none_chosen() {
                let (model, solution, word) =
                    build_test_fixtures("cat oven mutch cab abs cap", vec!["oven", "mutch"], "cat");
                assert_eq!(
                    word_score_new_pairs(&word, &solution.unchosen_pairs(&model)),
                    2
                );
            }

            #[test]
            fn word_score_new_pairs_one_chosen() {
                let (model, solution, word) =
                    build_test_fixtures("cat cab abs cap", vec!["cab", "abs"], "cat");
                assert_eq!(
                    word_score_new_pairs(&word, &solution.unchosen_pairs(&model)),
                    1
                );
            }

            #[test]
            fn word_score_new_pairs_empty_solution() {
                let (model, solution, word) =
                    build_test_fixtures("cat oven mutch cab abs cap", Vec::new(), "cat");
                assert_eq!(
                    word_score_new_pairs(&word, &solution.unchosen_pairs(&model)),
                    2
                );
            }

            #[test]
            fn word_score_new_pairs_repeated_pairs() {
                let test_word = "bananananana";
                let (model, solution, word) = build_test_fixtures(test_word, Vec::new(), test_word);
                assert_eq!(
                    word_score_new_pairs(&word, &solution.unchosen_pairs(&model)),
                    3
                );
            }

            #[test]
            fn empty() {
                let model = Model::new(io::empty()).expect("reading empty should not fail");
                assert_eq!(greedy_most_valuable_word(&model), Ok(Solution::default()));
            }

            #[test]
            fn one_word() {
                let (model, expected_solution) = build_test_model_and_solution("cat", vec!["cat"]);
                let solution = greedy_most_valuable_word(&model);
                assert_eq!(solution, Ok(expected_solution));
            }

            #[test]
            fn multiple() {
                let (model, expected_solution) = build_test_model_and_solution(
                    "cat abs hutch oven cab cap much coven",
                    vec!["coven", "hutch", "abs", "cap", "cat", "much"],
                );
                let solution =
                    greedy_most_valuable_word(&model).expect("test should produce solution");
                assert_eq!(solution, expected_solution);
            }

            #[test]
            fn find_best_word_word_score_new_pairs_lexical_choice() {
                let (model, solution, word) = build_test_fixtures("cat abs cab", Vec::new(), "abs");
                let best_word = find_best_word(
                    &word_score_new_pairs,
                    &solution.unchosen_words(&model),
                    &solution.unchosen_pairs(&model),
                );
                assert_eq!(best_word, Some(word));
            }

            #[test]
            fn find_best_word_word_score_new_pairs_high_score() {
                let (model, solution, word) =
                    build_test_fixtures("cat abs cab", vec!["abs"], "cat");
                let best_word = find_best_word(
                    &word_score_new_pairs,
                    &solution.unchosen_words(&model),
                    &solution.unchosen_pairs(&model),
                );
                assert_eq!(best_word, Some(word));
            }

            #[test]
            fn find_best_word_word_score_new_pairs_shortest() {
                let (model, solution, word) =
                    build_test_fixtures("cat abs cab carbs carb", vec!["cat", "abs"], "carb");
                let best_word = find_best_word(
                    &word_score_new_pairs,
                    &solution.unchosen_words(&model),
                    &solution.unchosen_pairs(&model),
                );
                assert_eq!(best_word, Some(word));
            }
        }
    }
}
