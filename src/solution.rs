//! A representation of a character pair problem solution.

use crate::text_model::{CharacterPair, Model, Word};
use std::{
    collections::HashSet,
    fmt::{self, Display, Formatter},
    rc::Rc,
};

#[derive(Default, Debug)]
/// A collection of `text_model::Words` representing a solution to the character pair problem.
///
/// A solution is not necessarily complete.
pub struct Solution {
    words: HashSet<Rc<Word>>,
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut repr = String::new();
        let mut first = true;

        for word in self.words.iter() {
            if !first {
                repr.push(' ');
            } else {
                first = false;
            };

            repr.push_str(&word.to_string());
        }

        write!(f, "{repr}")
    }
}

impl Solution {
    /// Adds a `Word` to the solution.
    pub fn add_word(&mut self, word: Rc<Word>) -> bool {
        self.words.insert(word)
    }

    /// Returns `true` if a `Word` is part of the solution, `false` if not.
    pub fn contains_word(&self, word: &Rc<Word>) -> bool {
        self.words.contains(word)
    }

    /// Returns a HashSet containing all the CharacterPairs covered by the solution.
    pub fn pairs(&self) -> HashSet<Rc<CharacterPair>> {
        let mut pairs = HashSet::new();
        for word in self.words.iter() {
            for pair in word.pairs() {
                pairs.insert(pair.clone());
            }
        }
        pairs
    }

    /// Returns true if the Solution is complete for a given Model.
    ///
    /// Checks that the solution covers all CharacterPairs contained within the Model.
    pub fn is_complete(&self, model: &Model) -> bool {
        model.pairs().difference(&self.pairs()).count() == 0
    }

    /// Returns true if the Solution is valid for a given Model.
    ///
    /// Checks that the solution is complete and uses only Words from the Model.
    pub fn is_valid(&self, model: &Model) -> bool {
        self.words.difference(model.words()).count() == 0 && self.is_complete(model)
    }

    /// Returns a Hashset of the Words in the solution.
    pub fn words(&self) -> &HashSet<Rc<Word>> {
        &self.words
    }
}

#[cfg(test)]
pub fn create_test_solution(model: &Model, words: Vec<&str>) -> Solution {
    let mut solution = Solution::default();

    for word in words.into_iter() {
        match model.words().iter().find(|&x| word == x.to_string()) {
            Some(x) => {
                solution.add_word(x.clone());
            }
            None => {
                panic!("Could not find {word:?} in model: {model:?}");
            }
        }
    }

    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_model::Model;

    #[test]
    fn display_empty() {
        let solution = Solution::default();
        assert_eq!(solution.to_string(), "");
    }

    #[test]
    fn display_one_word() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cab"]);

        assert_eq!(solution.to_string(), "cab");
    }

    #[test]
    fn display_words() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cat", "abs"]);

        let repr = solution.to_string();
        assert!(
            repr == "cat abs" || repr == "abs cat",
            "display created invalid output: {repr:?}"
        );
    }

    #[test]
    fn add_word() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");
        assert!(solution.add_word(word));
    }

    #[test]
    fn add_words() {
        let mut solution = Solution::default();
        let model = Model::build_test_model("cat abs hutch oven cab cap much coven");

        for word in model.words() {
            assert!(solution.add_word(word.clone()));
        }
    }

    #[test]
    fn add_duplicate() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");

        assert!(solution.add_word(word.clone()));
        assert!(!solution.add_word(word.clone()));
    }

    #[test]
    fn contains_word_empty() {
        let solution = Solution::default();
        let word = Word::build_test_word("cat");
        assert!(!solution.contains_word(&word));
    }

    #[test]
    fn contains_word_one_word() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");
        solution.add_word(word.clone());
        assert!(solution.contains_word(&word));
    }

    #[test]
    fn contains_word_multiple_words() {
        let mut solution = Solution::default();
        let model = Model::build_test_model("cat abs hutch oven cab cap much coven");

        for word in model.words() {
            solution.add_word(word.clone());
        }

        for word in model.words() {
            assert!(solution.contains_word(&word));
        }
    }

    #[test]
    fn contains_word_duplicate() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");

        solution.add_word(word.clone());
        solution.add_word(word.clone());
        assert!(solution.contains_word(&word));
    }

    #[test]
    fn pairs_empty() {
        let solution = Solution::default();
        assert_eq!(
            solution.pairs().len(),
            0,
            "Solution pairs was not empty: {solution:?}"
        );
    }

    #[test]
    fn pairs_one_word() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");
        solution.add_word(word.clone());

        assert_eq!(
            solution.pairs(),
            word.pairs().iter().map(|x| x.clone()).collect()
        );
    }

    #[test]
    fn pairs_multiple_words() {
        let model = Model::build_test_model("cat abs hutch oven cab cap much coven");
        let solution = create_test_solution(&model, vec!["cat", "abs", "oven", "much"]);
        let expected: HashSet<Rc<CharacterPair>> = vec![
            ('c', 'a'),
            ('a', 't'),
            ('a', 'b'),
            ('b', 's'),
            ('o', 'v'),
            ('v', 'e'),
            ('e', 'n'),
            ('m', 'u'),
            ('u', 'c'),
            ('c', 'h'),
        ]
        .into_iter()
        .map(|x| Rc::new(x))
        .collect();

        assert_eq!(solution.pairs(), expected);
    }

    #[test]
    fn pairs_duplicates() {
        let model = Model::build_test_model("cat abs hutch oven cab cap much coven");
        let solution = create_test_solution(&model, vec!["cat", "cab"]);
        let expected: HashSet<Rc<CharacterPair>> = vec![('c', 'a'), ('a', 't'), ('a', 'b')]
            .into_iter()
            .map(|x| Rc::new(x))
            .collect();

        assert_eq!(solution.pairs(), expected);
    }

    #[test]
    fn is_valid_empty_complete() {
        let solution = Solution::default();
        let model = Model::default();
        assert!(solution.is_valid(&model));
    }

    #[test]
    fn is_valid_empty_incomplete() {
        let solution = Solution::default();
        let model = Model::build_test_model("cat");
        assert!(!solution.is_valid(&model));
    }

    #[test]
    fn is_valid_emmpty_extra_word() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");
        solution.add_word(word);
        let model = Model::default();
        assert!(!solution.is_valid(&model));
    }

    #[test]
    fn is_valid_true() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cat", "abs"]);
        assert!(solution.is_valid(&model));
    }

    #[test]
    fn is_valid_false() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cab", "abs"]);
        assert!(!solution.is_valid(&model));
    }

    #[test]
    fn is_valid_overcomplete() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cat", "abs", "cab"]);
        assert!(solution.is_valid(&model));
    }

    #[test]
    fn is_valid_extra() {
        let model = Model::build_test_model("cat abs cab");
        let mut solution = create_test_solution(&model, vec!["cat", "abs", "cab"]);
        let word = Word::build_test_word("oven");
        solution.add_word(word);
        assert!(!solution.is_valid(&model));
    }

    #[test]
    fn is_complete_empty_complete() {
        let solution = Solution::default();
        let model = Model::default();
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_empty_incomplete() {
        let solution = Solution::default();
        let model = Model::build_test_model("cat");
        assert!(!solution.is_complete(&model));
    }

    #[test]
    fn is_complete_emmpty_extra_word() {
        let mut solution = Solution::default();
        let word = Word::build_test_word("cat");
        solution.add_word(word);
        let model = Model::default();
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_true() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cat", "abs"]);
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_false() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cab", "abs"]);
        assert!(!solution.is_complete(&model));
    }

    #[test]
    fn is_complete_overcomplete() {
        let model = Model::build_test_model("cat abs cab");
        let solution = create_test_solution(&model, vec!["cat", "abs", "cab"]);
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_extra() {
        let model = Model::build_test_model("cat abs cab");
        let mut solution = create_test_solution(&model, vec!["cat", "abs", "cab"]);
        let word = Word::build_test_word("oven");
        solution.add_word(word);
        assert!(solution.is_complete(&model));
    }
}
