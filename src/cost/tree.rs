//! Persistent segment tree.

use std::ops::RangeInclusive;

use ndarray::ArrayView1;
#[cfg(feature = "rayon")]
use rayon::slice::ParallelSliceMut as _;

/// Persistent segment tree for finding the K-th smallest value datastructure.
pub struct KthSmallestTree {
    /// Each root is a different version.
    roots: Vec<u32>,
    /// Total map of all node counts.
    ///
    /// We keep this separate from the nodes as a performance optimization, since it will only be accessed on the left children in the `kth()` implementation.
    counts: Vec<u32>,
    /// Total map of all node siblings.
    siblings: Vec<Node>,
    /// Sorted and unique values.
    sorted: Vec<f64>,
    /// Size of the original value array.
    len: u32,
}

impl KthSmallestTree {
    /// Construct the tree from a slice of values.
    #[inline]
    pub fn build(values: &ArrayView1<f64>) -> Self {
        assert!(values.len() < u32::MAX as usize - 1, "Input array too big");

        let roots = Vec::with_capacity(values.len());

        // Allocate at least n * log2(n) nodes plus a root
        let total_estimate = values.len() * values.len().next_power_of_two().ilog2() as usize + 1;
        let siblings = Vec::with_capacity(total_estimate);
        let counts = Vec::with_capacity(total_estimate);

        let len = values.len() as u32;

        let mut sorted = values.to_vec();
        // Sort the values
        #[cfg(feature = "rayon")]
        sorted.par_sort_unstable_by(f64::total_cmp);
        #[cfg(not(feature = "rayon"))]
        sorted.sort_unstable_by(f64::total_cmp);

        // Remove duplicates
        sorted.dedup();

        let mut this = Self {
            roots,
            siblings,
            counts,
            len,
            sorted: sorted.clone(),
        };

        // Build the zero version of the tree
        this.siblings.push(Node {
            left_index: 0,
            right_index: 0,
        });
        this.counts.push(0);
        this.roots.push(0);

        // Get each index
        let indices: Vec<u32> = values
            .iter()
            .map(|value| {
                // Lookup the index of the value but make it one-based
                sorted
                    .binary_search_by(|sorted_value| f64::total_cmp(sorted_value, value))
                    .unwrap_or_default()
                    .saturating_add(1) as u32
            })
            .collect();

        // Add each value index as a version update
        for index in indices {
            // Use one based indexing
            let root = this.insert(
                *this.roots.last().expect("Building root failed"),
                1..=len,
                index,
            );
            this.roots.push(root);
        }

        this
    }

    /// Find the K-th element.
    pub fn kth(&self, range: RangeInclusive<usize>, mut kth: usize) -> f64 {
        // Get the root node at the end
        let mut current_node = &self.siblings[self.roots[*range.end() + 1] as usize];
        // Get the root node at the start
        let mut previous_node = &self.siblings[self.roots[*range.start()] as usize];

        // Indices range to look for
        let mut start = 1_u32;
        let mut end = self.len;

        // Walk until item found
        while start != end {
            // Difference of sizes of the left nodes
            let left_size = {
                let current_left_count = self.counts[current_node.left_index as usize];
                let previous_left_count = self.counts[previous_node.left_index as usize];

                (current_left_count as usize).saturating_sub(previous_left_count as usize)
            };

            let mid = start.midpoint(end);

            // Find the offset point to go left or right
            if kth <= left_size {
                current_node = &self.siblings[current_node.left_index as usize];
                previous_node = &self.siblings[previous_node.left_index as usize];

                // start..=mid
                end = mid;
            } else {
                current_node = &self.siblings[current_node.right_index as usize];
                previous_node = &self.siblings[previous_node.right_index as usize];

                // mid+1..=end
                start = mid + 1;

                // Move to the right region
                kth -= left_size;
            }
        }

        // Leaf found
        self.sorted[start as usize - 1]
    }

    /// Recursive implementation of creating a new version.
    fn insert(&mut self, current_index: u32, range: RangeInclusive<u32>, update_index: u32) -> u32 {
        debug_assert!(update_index >= *range.start(), "{update_index} {range:?}");
        debug_assert!(update_index <= *range.end(), "{update_index} {range:?}");

        let current_index = current_index as usize;

        // Copy the previous node
        let mut node = self.siblings[current_index];
        let mut count = self.counts[current_index];
        count += 1;

        // If narrowed down to a leaf, push a new node and return it
        if range.start() == range.end() {
            let index = self.siblings.len() as u32;
            self.siblings.push(node);
            self.counts.push(count);

            return index;
        }

        // Traverse the tree
        let mid = range.start().midpoint(*range.end());

        // Update the two branches of a node
        if update_index <= mid {
            // Update the left half
            node.left_index = self.insert(node.left_index, *range.start()..=mid, update_index);
        } else {
            // Update the right half
            node.right_index =
                self.insert(node.right_index, (mid + 1)..=*range.end(), update_index);
        };

        // Push the node
        let index = self.siblings.len() as u32;
        self.siblings.push(node);
        self.counts.push(count);

        index
    }
}

/// Persistent segment tree node.
#[derive(Clone, Copy)]
pub struct Node {
    /// Left child.
    left_index: u32,
    /// Right child.
    right_index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K-th smallest number.
    #[test]
    fn kth() {
        let input = ndarray::aview1(&[3.5, 1.2, 4.8, 2.1, 5.0, 1.2]);
        let tree = KthSmallestTree::build(&input);

        // Sort the values over the whole range and use that to ensure it works
        let mut sorted = input.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.into_iter().enumerate().for_each(|(index, value)| {
            let range = 0..=(input.len() - 1);
            let kth = index + 1;
            assert_eq!(
                tree.kth(range.clone(), kth),
                value,
                "{input:?} {range:?} K-th {kth}",
            );
        });

        // Check different ranges
        assert_eq!(tree.kth(2..=4, 1), 2.1);
        assert_eq!(tree.kth(2..=4, 2), 4.8);
        assert_eq!(tree.kth(2..=4, 3), 5.0);
    }
}
