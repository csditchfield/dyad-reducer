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
    ConsistencyError,
}

/// Creates a `Solution` from a `Model` by choosing all words in the `Model`.
pub fn whole_text(model: &Model) -> Result<Solution, SolverError> {
    Ok(Solution::new(model.words().clone()))
}

fn word_score_new_pairs(word: &Word, covered_pairs: &HashSet<Rc<CharacterPair>>) -> usize {
    let mut duplicates = HashSet::new();
    let mut score = 0;
    for pair in word.pairs() {
        if !covered_pairs.contains(pair) && !duplicates.contains(pair) {
            score += 1;
            duplicates.insert(pair);
        }
    }
    score
}

/// Creates a `Solution` from a `text_model::Model` by repeatedly choosing
/// the most valuable `Word`.
///
/// The value of a word is determined by how many unchosen `CharacterPairs`
/// it contains.
pub fn greedy_most_valuable_word(model: &Model) -> Result<Solution, SolverError> {
    let mut solution = Solution::default();

    while !solution.is_complete(model) {
        let covered_pairs = solution.pairs();

        let mut unchosen_words = model.words().difference(solution.words());
        let mut best_word: &Rc<Word> = match unchosen_words.next() {
            Some(x) => x,
            None => {
                return Err(SolverError::new(
                    String::from("Solution is incomplete, but there are no more unchosen words."),
                    ErrorKind::ConsistencyError,
                    None,
                ))
            }
        };
        let mut high_score = word_score_new_pairs(best_word, &covered_pairs);

        for word in unchosen_words {
            let score = word_score_new_pairs(word, &covered_pairs);
            if (score > high_score) || (score == high_score && word.len() < best_word.len()) {
                high_score = score;
                best_word = word;
            };
        }

        solution.add_word(best_word.clone());
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::create_test_solution;
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

    mod greedy_most_valuable_word {
        use super::*;

        #[test]
        fn word_score_new_pairs_all_chosen() {
            let model = Model::build_test_model("cat at ca");
            let solution = create_test_solution(&model, vec!["at", "ca"]);
            let word = model
                .words()
                .iter()
                .find(|x| x.to_string() == "cat")
                .expect("model should contain test word");
            assert_eq!(word_score_new_pairs(&word, &solution.pairs()), 0);
        }

        #[test]
        fn word_score_new_pairs_none_chosen() {
            let model = Model::build_test_model("cat oven mutch cab abs cap");
            let solution = create_test_solution(&model, vec!["oven", "mutch"]);
            let word = model
                .words()
                .iter()
                .find(|x| x.to_string() == "cat")
                .expect("model should contain test word");
            assert_eq!(word_score_new_pairs(&word, &solution.pairs()), 2);
        }

        #[test]
        fn word_score_new_pairs_one_chosen() {
            let model = Model::build_test_model("cat cab abs cap");
            let solution = create_test_solution(&model, vec!["cab", "abs"]);
            let word = model
                .words()
                .iter()
                .find(|x| x.to_string() == "cat")
                .expect("model should contain test word");
            assert_eq!(word_score_new_pairs(&word, &solution.pairs()), 1);
        }

        #[test]
        fn word_score_new_pairs_empty_solution() {
            let model = Model::build_test_model("cat oven mutch cab abs cap");
            let solution = Solution::default();
            let word = model
                .words()
                .iter()
                .find(|x| x.to_string() == "cat")
                .expect("model should contain test word");
            assert_eq!(word_score_new_pairs(&word, &solution.pairs()), 2);
        }

        #[test]
        fn word_score_new_pairs_repeated_pairs() {
            let test_word = "bananananana";
            let model = Model::build_test_model(test_word);
            let solution = Solution::default();
            let word = model
                .words()
                .iter()
                .find(|x| x.to_string() == test_word)
                .expect("model should contain test word");
            assert_eq!(word_score_new_pairs(&word, &solution.pairs()), 3);
        }

        #[test]
        fn empty() {
            let model = Model::new(io::empty()).expect("reading empty should not fail");
            assert!(greedy_most_valuable_word(&model)
                .expect("test should produce solution")
                .is_valid(&model));
        }

        #[test]
        fn one_word() {
            let model = Model::build_test_model("cat");
            let solution = greedy_most_valuable_word(&model).expect("test should produce solution");
            assert!(solution.is_valid(&model));
            assert_eq!(solution.words().iter().count(), 1);
        }

        #[test]
        fn multiple() {
            let model = Model::build_test_model("cat abs hutch oven cab cap much coven");
            let solution = greedy_most_valuable_word(&model).expect("test should produce solution");
            assert!(solution.is_valid(&model));
        }
    }
}
