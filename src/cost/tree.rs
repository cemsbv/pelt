//! Persistent segment tree.

use std::ops::RangeInclusive;

use ndarray::ArrayView1;

/// Persistent segment tree for finding the K-th smallest value datastructure.
pub struct KthSmallestTree {
    /// Each root is a different version.
    roots: Vec<u32>,
    /// Total map of all nodes.
    nodes: Vec<Node>,
    /// Sorted and unique values.
    sorted: Vec<f64>,
    /// Size of the original value array.
    len: u32,
}

impl KthSmallestTree {
    /// Construct the tree from a slice of values.
    #[inline]
    pub fn build(values: &ArrayView1<f64>) -> Self {
        let roots = Vec::new();
        let nodes = Vec::with_capacity(4 * values.len());
        let len = values.len() as u32;

        let mut sorted = values.to_vec();
        // Sort the values
        sorted.sort_by(f64::total_cmp);
        // Remove duplicates
        sorted.dedup();

        let mut this = Self {
            roots,
            nodes,
            len,
            sorted: sorted.clone(),
        };

        // Build the zero version of the tree
        this.nodes.push(Node {
            count: 0,
            left_index: 0,
            right_index: 0,
        });
        this.roots.push(0);

        // Add each value as a version update
        for value in values.iter() {
            // Lookup the index of the value but make it one-based
            let index = sorted
                .binary_search_by(|sorted_value| f64::total_cmp(sorted_value, value))
                .unwrap_or_default()
                + 1;

            // Use one based indexing
            let root = this.insert(
                *this.roots.last().expect("Building root failed"),
                1..=len,
                index as u32,
            );
            this.roots.push(root);
        }

        this
    }

    /// Find the K-th element.
    #[inline]
    pub fn kth(&self, range: RangeInclusive<usize>, kth: usize) -> f64 {
        // Find the index into the sorted array
        let index = self.kth_impl(
            self.roots[*range.end() + 1],
            self.roots[*range.start()],
            1..=self.len,
            kth as i32,
        );

        self.sorted[index as usize - 1]
    }

    /// Find the K-th element in the sorted array.
    fn kth_impl(
        &self,
        current_index: u32,
        previous_index: u32,
        range: RangeInclusive<u32>,
        kth: i32,
    ) -> u32 {
        // Return item if narrowed down
        if range.start() == range.end() {
            return *range.start();
        }

        let current_node = self.nodes[current_index as usize].clone();
        let previous_node = self.nodes[previous_index as usize].clone();

        // Difference of sizes of the left nodes
        let left_size = self.nodes[current_node.left_index as usize].count
            - self.nodes[previous_node.left_index as usize].count;

        let mid = range.start().midpoint(*range.end());

        // Find the offset point to go left or right
        if kth <= left_size {
            self.kth_impl(
                current_node.left_index,
                previous_node.left_index,
                Self::left_range(&range, mid),
                kth,
            )
        } else {
            self.kth_impl(
                current_node.right_index,
                previous_node.right_index,
                Self::right_range(&range, mid),
                kth - left_size,
            )
        }
    }

    /// Recursive implementation of creating a new version.
    fn insert(&mut self, current_index: u32, range: RangeInclusive<u32>, update_index: u32) -> u32 {
        debug_assert!(update_index >= *range.start(), "{update_index} {range:?}");
        debug_assert!(update_index <= *range.end(), "{update_index} {range:?}");

        // Copy the previous node
        let mut node = self.nodes[current_index as usize].clone();
        node.count += 1;

        // If narrowed down to a leaf, push a new node and return it
        if range.start() == range.end() {
            let index = self.nodes.len() as u32;
            self.nodes.push(node);

            return index;
        }

        // Traverse the tree
        let mid = range.start().midpoint(*range.end());

        // Update the two branches of a node
        if update_index <= mid {
            // Update the left half
            node.left_index =
                self.insert(node.left_index, Self::left_range(&range, mid), update_index);
        } else {
            // Update the right half
            node.right_index = self.insert(
                node.right_index,
                Self::right_range(&range, mid),
                update_index,
            );
        };

        // Push the node
        let index = self.nodes.len() as u32;
        self.nodes.push(node);

        index
    }

    /// Get the left ranges separated by the midpoint.
    #[inline]
    const fn left_range(range: &RangeInclusive<u32>, mid: u32) -> RangeInclusive<u32> {
        *range.start()..=mid
    }

    /// Get the right ranges separated by the midpoint.
    #[inline]
    const fn right_range(range: &RangeInclusive<u32>, mid: u32) -> RangeInclusive<u32> {
        (mid + 1)..=*range.end()
    }
}

/// Persistent segment tree node.
#[derive(Clone)]
pub struct Node {
    /// Value of the node.
    count: i32,
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
