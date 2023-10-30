# Dyad Reducer

This program is a collection of algorithms related to the _character-pair problem_. Given a source text composed of words separated by whitespace characters, a solution to the character-pair problem is a text composed of words from the source that includes every pair of characters that appear in that source text. This repository includes solvers that attempt to find minimal solutions by various means, as well as other utilities such as a solution checker.

## Running

This program can be built from source using [Cargo](https://doc.rust-lang.org/cargo/index.html), Rust's package manager.

`cargo run` is the easiest way to get started, flags for the dyad reducer are passed after two dashes. e.g. `cargo run -- ./example_input.txt`

A full list of available commands can be produced by using the `--help` argument.

## Solvers

Here are descriptions of the implemented solvers:

### Greedy Most Valuable Word

This greedy algorithm creates a solution by repeatedly scanning the list of unchosen words and choosing the shortest word that contains the largest number of unchosen character pairs.

## Contributing

Contributions are very welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Dyad Reducer is distributed under the terms of the Apache License (Version 2.0). See [LICENSE.txt](LICENSE.txt) for details.
