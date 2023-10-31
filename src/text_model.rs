//! Model of a text in terms of character pairs.
//!
//! Provides utilities for creating a model from string slices
//! and anything that implements BufRead.

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
    io::BufRead,
    rc::Rc,
};

use crate::error::{ErrorKind, SolverError};

/// A Word.
///
/// A word is a sequence of one or more character pairs, and
/// may have multiple copies of the individual pairs.
///
/// A Word does not have meaning outside of the context of a
/// Model, hence typically they are created by Models.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Word {
    pairs: Vec<Rc<CharacterPair>>,
}

impl Display for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut repr = self.pairs.iter().fold(String::new(), |mut acc, x| {
            acc.push(x.0);
            acc
        });

        match self.pairs.last() {
            Some(x) => repr.push(x.1),
            None => (),
        };

        write!(f, "{repr}")
    }
}

impl Word {
    /// Returns a reference to the character pairs making up the word.
    pub fn pairs(&self) -> &Vec<Rc<CharacterPair>> {
        &self.pairs
    }

    /// Returns the total character length of the word.
    pub fn len(&self) -> usize {
        let pairs = self.pairs.len();
        if pairs == 0 {
            0
        } else {
            pairs + 1
        }
    }

    #[cfg(test)]
    /// Builds a word from a string.
    ///
    /// The word will be internally consistent, sharing a single
    /// CharacterPair with multiple Rcs.
    /// It will have no relation to any CharacterPairs from any
    /// Model or source text however.
    ///
    /// # Example
    /// ```
    /// let word = Word::build_test_word("cat");
    /// assert_eq!(word.to_string(), "cat");
    /// ```
    pub fn build_test_word(value: &str) -> Rc<Self> {
        let mut pairs: Vec<Rc<CharacterPair>> = Vec::default();
        let mut seen: HashSet<Rc<CharacterPair>> = HashSet::default();
        let mut prev = None;

        for c in value.chars() {
            match prev {
                Some(p) => {
                    let pair = Rc::new((p, c));
                    match seen.get(&pair) {
                        Some(p) => {
                            pairs.push(p.clone());
                        }
                        None => {
                            seen.insert(pair.clone());
                            pairs.push(pair);
                        }
                    };
                }
                None => {}
            };
            prev = Some(c);
        }
        Rc::new(Word {
            pairs: pairs.into_iter().collect(),
        })
    }
}

/// A Pair of Characters.
pub type CharacterPair = (char, char);

/// A Model of a source text in terms of character pairs.
///
/// A text is composed of zero or more words which are seperated by whitespace
/// and composed of one or more pairs of characters. The model contains only
/// one instance of any word or character pair, with words sharing ownership
/// of character pairs.
///
/// If you have a `Word`, finding the contained character pairs is
/// accomplished with the word's `pairs()` method. If you have a
/// `CharacterPair`, finding which words contain the pair is
/// performed by the `Model`'s `get_pair_words()` method.
///
/// # Example
/// ``` rust
/// use std::rc::Rc;
/// use dyad_reducer::text_model::Model;
///
/// # fn main() -> Result<(), dyad_reducer::error::SolverError> {
/// let text = "Cats are great!\ncats are cute in caps.";
/// let model = Model::try_from(text)?;
///
/// // "are" is part of the model only once.
/// assert_eq!(model.words().len(), text.split_whitespace().count()-1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct Model {
    words: HashSet<Rc<Word>>,
    pairs: HashSet<Rc<CharacterPair>>,
    pair_mapping: HashMap<Rc<CharacterPair>, HashSet<Rc<Word>>>,
}

impl TryFrom<&str> for Model {
    type Error = SolverError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Model::try_from(value.as_bytes())
    }
}

impl TryFrom<&[u8]> for Model {
    type Error = SolverError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Model::new(value)
    }
}

impl Model {
    /// Builds a `Model` from a type that implements `std::io::BufRead`.
    ///
    /// Raises any `io::Error` that results from reading the `BufRead`
    pub fn new(reader: impl BufRead) -> Result<Model, SolverError> {
        let mut pairs: HashSet<Rc<CharacterPair>> = HashSet::new();
        let mut words: HashSet<Rc<Word>> = HashSet::new();
        let mut pair_mapping: HashMap<Rc<CharacterPair>, HashSet<Rc<Word>>> = HashMap::new();
        let mut words_seen: HashSet<String> = HashSet::new();

        for outcome in reader.lines() {
            let line = match outcome {
                Ok(x) => x,
                Err(e) => {
                    return Err(SolverError::new(
                        String::from("Could not read file."),
                        ErrorKind::FileReadingError,
                        Some(Box::new(e)),
                    ))
                }
            };
            for str_word in line.split_whitespace() {
                if str_word.len() >= 2 && !words_seen.contains(str_word) {
                    words_seen.insert(String::from(str_word));
                    let mut word_pairs = Vec::new();
                    let mut prev_char = None;
                    for c in str_word.chars() {
                        match prev_char {
                            Some(p) => {
                                let new_pair = Rc::new((p, c));
                                let pair = match pairs.get(&new_pair) {
                                    Some(p) => p.clone(),
                                    None => {
                                        pairs.insert(new_pair.clone());
                                        new_pair
                                    }
                                };
                                word_pairs.push(pair);
                            }
                            None => (),
                        };
                        prev_char = Some(c);
                    }
                    let word = Rc::new(Word { pairs: word_pairs });
                    words.insert(word.clone());

                    for pair in word.pairs.iter() {
                        match pair_mapping.get_mut(pair) {
                            Some(x) => {
                                x.insert(word.clone());
                            }
                            None => {
                                let map = vec![word.clone()].into_iter().collect();
                                pair_mapping.insert(pair.clone(), map);
                            }
                        };
                    }
                }
            }
        }

        Ok(Model {
            pairs,
            words,
            pair_mapping,
        })
    }

    /// Return a reference to the `Word`s in the text.
    pub fn words(&self) -> &HashSet<Rc<Word>> {
        &self.words
    }

    /// Return a reference to the `CharacterPair`s in the text.
    pub fn pairs(&self) -> &HashSet<Rc<CharacterPair>> {
        &self.pairs
    }

    /// Get the `Word`s that contain a given `CharacterPair`.
    ///
    /// Returns `None` if the `CharacterPair` is not in the `Model`.
    /// Returns a reference to a `HashSet` of `Word`s within `Rc`s.
    pub fn get_pair_words(&self, pair: &CharacterPair) -> Option<&HashSet<Rc<Word>>> {
        self.pair_mapping.get(pair)
    }

    #[cfg(test)]
    /// Constructs a `Model` from a string slice.
    ///
    /// # Panics
    /// Will panic if there is an error `BufRead`ing from the slice.
    pub fn build_test_model(input: &str) -> Self {
        Model::try_from(input).expect("test model creation should not fail because reading a byte array should never produce an error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod word_tests {
        use super::*;

        #[test]
        fn display_empty() {
            let word = Word { pairs: Vec::new() };
            assert_eq!(word.to_string(), "");
        }

        #[test]
        fn display_one() {
            let word = Word {
                pairs: vec![Rc::new(('a', 'b'))],
            };
            assert_eq!(word.to_string(), "ab");
        }

        #[test]
        fn display_long() {
            let pairs = vec![
                Rc::new(('A', 'd')),
                Rc::new(('d', 'v')),
                Rc::new(('v', 'a')),
                Rc::new(('a', 'n')),
                Rc::new(('n', 'c')),
                Rc::new(('c', 'e')),
                Rc::new(('e', 'd')),
            ];
            let word = Word { pairs };

            assert_eq!(word.to_string(), "Advanced");
        }

        #[test]
        fn pairs_empty() {
            let word = Word {
                pairs: Vec::default(),
            };
            assert_eq!(*word.pairs(), Vec::default());
        }

        #[test]
        fn pairs_one() {
            let pairs = vec![Rc::new(('a', 'b'))];
            let word = Word {
                pairs: pairs.clone(),
            };
            assert_eq!(*word.pairs(), pairs);
        }

        #[test]
        fn pairs_long() {
            let pairs = vec![
                Rc::new(('A', 'd')),
                Rc::new(('d', 'v')),
                Rc::new(('v', 'a')),
                Rc::new(('a', 'n')),
                Rc::new(('n', 'c')),
                Rc::new(('c', 'e')),
                Rc::new(('e', 'd')),
            ];
            let word = Word {
                pairs: pairs.clone(),
            };

            assert_eq!(*word.pairs(), pairs);
        }

        #[test]
        fn display_duplicates() {
            let an = Rc::new(('a', 'n'));
            let na = Rc::new(('n', 'a'));
            let pairs = vec![
                Rc::new(('b', 'a')),
                an.clone(),
                na.clone(),
                an.clone(),
                na.clone(),
            ];
            let word = Word { pairs };

            assert_eq!(word.to_string(), "banana");
        }

        #[test]
        fn build_test_word_empty() {
            let word = Word::build_test_word("");
            assert_eq!(word.pairs().len(), 0);
        }

        #[test]
        fn build_test_word() {
            let word = Word::build_test_word("banana");
            let ba = Rc::new(('b', 'a'));
            let an = Rc::new(('a', 'n'));
            let na = Rc::new(('n', 'a'));
            let expected_pairs = vec![ba, an.clone(), na.clone(), an, na];
            assert_eq!(*word.pairs(), expected_pairs);
        }

        #[test]
        fn len_empty() {
            let word = Word::build_test_word("");
            assert_eq!(word.len(), 0);
        }

        #[test]
        fn len_one_pair() {
            let word = Word::build_test_word("at");
            assert_eq!(word.len(), 2);
        }

        #[test]
        fn len_long() {
            let word = Word::build_test_word("honorificabilitudinitatibus");
            assert_eq!(word.len(), 27);
        }
    }

    mod model_tests {
        use super::*;
        use crate::test_reader::TestReader;
        use std::{
            error::Error,
            io::{self, BufReader},
        };

        fn verify_words(words: &HashSet<Rc<Word>>, expected: &HashSet<String>) {
            assert_eq!(
                words.len(),
                expected.len(),
                "Set sizes differ\nwords: {words:?}\n{expected:?}"
            );
            for word in words {
                assert!(
                    expected.contains(&word.to_string()),
                    "words contains unexpected word {word}"
                );
            }
        }

        fn verify_pairs(pairs: &HashSet<Rc<CharacterPair>>, expected: &HashSet<CharacterPair>) {
            assert_eq!(
                pairs.len(),
                expected.len(),
                "Set sizes differ\npairs: {pairs:?}\n{expected:?}"
            );
            for pair in pairs {
                assert!(
                    expected.contains(pair),
                    "pairs contains unexpected pair {pair:?}"
                );
            }
        }

        fn verify_pair_mapping(model: Model) {
            for word in model.words() {
                for pair in word.pairs() {
                    let mapping = model
                        .get_pair_words(pair)
                        .expect("Model missing mapping entry for pair: {pair:?}");
                    assert!(mapping.contains(word));
                }
            }
        }

        #[test]
        fn create_model_empty() {
            let input = BufReader::new(io::empty());
            let model = Model::new(input).expect("Model creation should not fail");
            assert_eq!(model.words().len(), 0);
            assert_eq!(model.pairs().len(), 0);
        }

        #[test]
        fn create_model_error() {
            let kind = io::ErrorKind::Other;
            let message = String::from("test error text");
            let input = BufReader::new(TestReader::new(
                "a bunch of text",
                Some((kind, message.clone())),
            ));
            let error =
                Model::new(input).expect_err("Model creation should fail when reading test reader");
            assert_eq!(*error.kind(), crate::error::ErrorKind::FileReadingError);
            let source = error.source().expect("error should have source io error.");
            assert!(source.to_string().contains(&message));
        }

        #[test]
        fn create_model_one_character() {
            let model = Model::build_test_model("a\n");
            let words = model.words();
            assert_eq!(words.len(), 0, "words is not empty. words: {words:?}");
            let pairs = model.pairs();
            assert_eq!(pairs.len(), 0, "pairs is not empty. pairs: {pairs:?}");
        }

        #[test]
        fn create_model_one_word() {
            let text = "an\n";
            let expected_words: HashSet<String> =
                text.split_whitespace().map(|x| String::from(x)).collect();
            let expected_pairs: HashSet<CharacterPair> = vec![('a', 'n')].into_iter().collect();
            let model = Model::build_test_model(text);
            verify_words(model.words(), &expected_words);
            verify_pairs(model.pairs(), &expected_pairs);
            verify_pair_mapping(model);
        }

        #[test]
        fn create_model_duplicate_pairs() {
            let text = "banana";
            let expected_words: HashSet<String> =
                text.split_whitespace().map(|x| String::from(x)).collect();
            let expected_pairs: HashSet<CharacterPair> = vec![('b', 'a'), ('a', 'n'), ('n', 'a')]
                .into_iter()
                .collect();
            let model = Model::build_test_model(text);
            verify_words(model.words(), &expected_words);
            verify_pairs(model.pairs(), &expected_pairs);
            verify_pair_mapping(model);
        }

        #[test]
        fn create_model_duplicate_words() {
            let text = "cat abs cat";
            let expected_words: HashSet<String> = vec!["cat", "abs"]
                .into_iter()
                .map(|x| String::from(x))
                .collect();
            let expected_pairs: HashSet<CharacterPair> =
                vec![('c', 'a'), ('a', 't'), ('a', 'b'), ('b', 's')]
                    .into_iter()
                    .collect();
            let model = Model::build_test_model(text);
            verify_words(model.words(), &expected_words);
            verify_pairs(model.pairs(), &expected_pairs);
            verify_pair_mapping(model);
        }

        #[test]
        fn create_model_multiple_lines() {
            let text = "cat abs\ncab \n  cap";
            let expected_words: HashSet<String> = vec!["cat", "abs", "cab", "cap"]
                .into_iter()
                .map(|x| String::from(x))
                .collect();
            let expected_pairs: HashSet<CharacterPair> =
                vec![('c', 'a'), ('a', 't'), ('a', 'b'), ('b', 's'), ('a', 'p')]
                    .into_iter()
                    .collect();
            let model = Model::build_test_model(text);
            verify_words(model.words(), &expected_words);
            verify_pairs(model.pairs(), &expected_pairs);
            verify_pair_mapping(model);
        }

        #[test]
        fn model_try_from_str() {
            let text = "cat abs\ncab \n  cap";
            let expected_words: HashSet<String> = vec!["cat", "abs", "cab", "cap"]
                .into_iter()
                .map(|x| String::from(x))
                .collect();
            let expected_pairs: HashSet<CharacterPair> =
                vec![('c', 'a'), ('a', 't'), ('a', 'b'), ('b', 's'), ('a', 'p')]
                    .into_iter()
                    .collect();
            match Model::try_from(text) {
                Ok(model) => {
                    verify_words(model.words(), &expected_words);
                    verify_pairs(model.pairs(), &expected_pairs);
                    verify_pair_mapping(model);
                }
                Err(e) => {
                    panic!("Expected model creation to succeed, got: {e:?} instead")
                }
            }
        }

        #[test]
        fn model_try_from_byte_array() {
            let text = "cat abs\ncab \n  cap";
            let expected_words: HashSet<String> = vec!["cat", "abs", "cab", "cap"]
                .into_iter()
                .map(|x| String::from(x))
                .collect();
            let expected_pairs: HashSet<CharacterPair> =
                vec![('c', 'a'), ('a', 't'), ('a', 'b'), ('b', 's'), ('a', 'p')]
                    .into_iter()
                    .collect();
            match Model::try_from(text.as_bytes()) {
                Ok(model) => {
                    verify_words(model.words(), &expected_words);
                    verify_pairs(model.pairs(), &expected_pairs);
                    verify_pair_mapping(model);
                }
                Err(e) => {
                    panic!("Expected model creation to succeed, got: {e:?} instead")
                }
            }
        }

        #[test]
        fn build_test_model() {
            let text = "cat abs\ncab \n  cap";
            let expected_words: HashSet<String> = vec!["cat", "abs", "cab", "cap"]
                .into_iter()
                .map(|x| String::from(x))
                .collect();
            let expected_pairs: HashSet<CharacterPair> =
                vec![('c', 'a'), ('a', 't'), ('a', 'b'), ('b', 's'), ('a', 'p')]
                    .into_iter()
                    .collect();
            let model = Model::build_test_model(text);
            verify_words(model.words(), &expected_words);
            verify_pairs(model.pairs(), &expected_pairs);
            verify_pair_mapping(model);
        }
    }
}
