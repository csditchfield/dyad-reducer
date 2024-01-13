//! A representation of a character pair problem solution.

use crate::text_model::{CharacterPair, Model, Word};
use std::{
    collections::HashSet,
    fmt::{self, Display, Formatter},
    rc::Rc,
};

#[derive(Clone, Default, Debug, Eq, PartialEq)]
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
    /// Create a Solution populated with the given words.
    pub fn new(words: HashSet<Rc<Word>>) -> Self {
        Self { words }
    }

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

    /// Returns a `Vec` of `Rc<CharacterPair>>` that are in model, but not self.
    pub fn unchosen_pairs(&self, model: &Model) -> HashSet<Rc<CharacterPair>> {
        model.pairs().difference(&self.pairs()).cloned().collect()
    }

    /// Returns a `Vec` of `Rc<CharacterPair>>` that are in model, but not self.
    pub fn unchosen_words(&self, model: &Model) -> HashSet<Rc<Word>> {
        model.words().difference(self.words()).cloned().collect()
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

    /// Returns the total character length of the solution.
    pub fn len(&self) -> usize {
        self.words().iter().map(|x| x.len()).sum::<usize>() + self.words().len().saturating_sub(1)
    }

    /// Returns true if the solution contains no words.
    pub fn is_empty(&self) -> bool {
        self.words().is_empty()
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
pub fn build_test_model_and_solution(
    model_text: &str,
    chosen_words: Vec<&str>,
) -> (Model, Solution) {
    let model = Model::build_test_model(model_text);
    let solution = create_test_solution(&model, chosen_words);
    (model, solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_model::Model;
    use std::io;

    #[test]
    fn new() {
        let text = "cat abs cab";
        let (model, solution) =
            build_test_model_and_solution(text, text.split_whitespace().collect());
        assert_eq!(solution.words(), model.words());
    }

    #[test]
    fn display_empty() {
        let solution = Solution::default();
        assert_eq!(solution.to_string(), "");
    }

    #[test]
    fn display_one_word() {
        let (_model, solution) = build_test_model_and_solution("cat abs cab", vec!["cab"]);
        assert_eq!(solution.to_string(), "cab");
    }

    #[test]
    fn display_words() {
        let (_model, solution) = build_test_model_and_solution("cat abs cab", vec!["cat", "abs"]);
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
        let (_model, solution) = build_test_model_and_solution(
            "cat abs hutch oven cab cap much coven",
            vec!["cat", "abs", "oven", "much"],
        );
        let expected: HashSet<Rc<CharacterPair>> = vec![
            CharacterPair::new('c', 'a'),
            CharacterPair::new('a', 't'),
            CharacterPair::new('a', 'b'),
            CharacterPair::new('b', 's'),
            CharacterPair::new('o', 'v'),
            CharacterPair::new('v', 'e'),
            CharacterPair::new('e', 'n'),
            CharacterPair::new('m', 'u'),
            CharacterPair::new('u', 'c'),
            CharacterPair::new('c', 'h'),
        ]
        .into_iter()
        .map(|x| Rc::new(x))
        .collect();

        assert_eq!(solution.pairs(), expected);
    }

    #[test]
    fn pairs_duplicates() {
        let (_model, solution) = build_test_model_and_solution(
            "cat abs hutch oven cab cap much coven",
            vec!["cat", "cab"],
        );
        let expected: HashSet<Rc<CharacterPair>> = vec![
            CharacterPair::new('c', 'a'),
            CharacterPair::new('a', 't'),
            CharacterPair::new('a', 'b'),
        ]
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
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["cat", "abs"]);
        assert!(solution.is_valid(&model));
    }

    #[test]
    fn is_valid_false() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["cab", "abs"]);
        assert!(!solution.is_valid(&model));
    }

    #[test]
    fn is_valid_overcomplete() {
        let (model, solution) =
            build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
        assert!(solution.is_valid(&model));
    }

    #[test]
    fn is_valid_extra() {
        let (model, mut solution) =
            build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
        solution.add_word(Word::build_test_word("oven"));
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
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["cat", "abs"]);
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_false() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["cab", "abs"]);
        assert!(!solution.is_complete(&model));
    }

    #[test]
    fn is_complete_overcomplete() {
        let (model, solution) =
            build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn is_complete_extra() {
        let (model, mut solution) =
            build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
        solution.add_word(Word::build_test_word("oven"));
        assert!(solution.is_complete(&model));
    }

    #[test]
    fn len_empty() {
        let solution = Solution::default();
        assert_eq!(solution.len(), 0);
    }

    #[test]
    fn len_one_word() {
        let (_model, solution) = build_test_model_and_solution("cat", vec!["cat"]);
        assert_eq!(solution.len(), 3);
    }

    #[test]
    fn len_words() {
        let (_model, solution) =
            build_test_model_and_solution("cat abs hutch", vec!["cat", "abs", "hutch"]);
        assert_eq!(solution.len(), 13);
    }

    #[test]
    fn len_words_to_string() {
        let (_model, solution) =
            build_test_model_and_solution("cat abs hutch", vec!["cat", "abs", "hutch"]);
        assert_eq!(solution.len(), solution.to_string().len());
    }

    #[test]
    fn is_empty_true() {
        let solution = Solution::default();
        assert!(solution.is_empty());
    }

    #[test]
    fn is_empty_false() {
        let mut solution = Solution::default();
        solution.add_word(Word::build_test_word("cat"));
        assert!(!solution.is_empty());
    }

    #[test]
    fn unchosen_pairs_both_empty() {
        let model = Model::new(io::empty()).expect("reading empty should not fail");
        let solution = Solution::default();
        assert_eq!(solution.unchosen_pairs(&model), HashSet::new());
    }

    #[test]
    fn unchosen_pairs_empty_model() {
        let test_model = Model::new(io::empty()).expect("reading empty should not fail");
        let (_, solution) = build_test_model_and_solution("cat", vec!["cat"]);
        assert_eq!(solution.unchosen_pairs(&test_model), HashSet::new());
    }

    #[test]
    fn unchosen_pairs_empty_solution() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", Vec::new());
        assert_eq!(solution.unchosen_pairs(&model), model.pairs().clone());
    }

    #[test]
    fn unchosen_pairs_match() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["cat", "abs"]);
        assert_eq!(solution.unchosen_pairs(&model), HashSet::new());
    }

    #[test]
    fn unchosen_pairs_extra_model() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["abs", "cab"]);
        assert_eq!(
            solution.unchosen_pairs(&model),
            HashSet::from([Rc::new(CharacterPair::new('a', 't'))])
        );
    }

    #[test]
    fn unchosen_pairs_extra_solution() {
        let (model, mut solution) = build_test_model_and_solution("cat abs", vec!["cat", "abs"]);
        solution.add_word(Word::build_test_word("red"));
        assert_eq!(solution.unchosen_pairs(&model), HashSet::new());
    }

    #[test]
    fn unchosen_pairs_extra_both() {
        let (model, mut solution) =
            build_test_model_and_solution("cat abs cab", vec!["cab", "abs"]);
        solution.add_word(Word::build_test_word("red"));
        assert_eq!(
            solution.unchosen_pairs(&model),
            HashSet::from([Rc::new(CharacterPair::new('a', 't'))])
        );
    }

    #[test]
    fn unchosen_words_both_empty() {
        let model = Model::new(io::empty()).expect("reading empty should not fail");
        let solution = Solution::default();
        assert_eq!(solution.unchosen_words(&model), HashSet::new());
    }

    #[test]
    fn unchosen_words_empty_model() {
        let test_model = Model::new(io::empty()).expect("reading empty should not fail");
        let (_, solution) = build_test_model_and_solution("cat", vec!["cat"]);
        assert_eq!(solution.unchosen_words(&test_model), HashSet::new());
    }

    #[test]
    fn unchosen_words_empty_solution() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", Vec::new());
        assert_eq!(solution.unchosen_words(&model), model.words().clone());
    }

    #[test]
    fn unchosen_words_match() {
        let (model, solution) =
            build_test_model_and_solution("cat abs cab", vec!["cat", "abs", "cab"]);
        assert_eq!(solution.unchosen_words(&model), HashSet::new());
    }

    #[test]
    fn unchosen_words_extra_model() {
        let (model, solution) = build_test_model_and_solution("cat abs cab", vec!["abs", "cab"]);
        assert_eq!(
            solution.unchosen_words(&model),
            HashSet::from([model.find_word_str("cat").expect("should contain \"cat\"")])
        );
    }

    #[test]
    fn unchosen_words_extra_solution() {
        let (model, mut solution) = build_test_model_and_solution("cat abs", vec!["cat", "abs"]);
        solution.add_word(Word::build_test_word("red"));
        assert_eq!(solution.unchosen_words(&model), HashSet::new());
    }

    #[test]
    fn unchosen_words_extra_both() {
        let (model, mut solution) =
            build_test_model_and_solution("cat abs cab", vec!["cab", "abs"]);
        solution.add_word(Word::build_test_word("red"));
        assert_eq!(
            solution.unchosen_words(&model),
            HashSet::from([model.find_word_str("cat").expect("should contain \"cat\"")])
        );
    }
}
