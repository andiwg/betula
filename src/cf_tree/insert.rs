//! Insertion logic for CFTree — macro-generated function pairs.
//!
//! A single shared template (`__insert_variant!`) is invoked twice — once for
//! point insertion and once for CF insertion. Only 4 points of divergence are
//! parameterized; the rest of the logic lives in one place.

use crate::cluster_feature::ClusterFeature;
use crate::distance::CFDistance;
use crate::types::Float;

use super::{CFNode, CFTree};

/// Overflow return — expands inside the generated function where
/// `new_leaf_entry` and the input parameter are in scope.
macro_rules! __overflow_ret {
  (point, $_:ident, $e:ident) => {
    Some($e)
  };
  (cf,    $p:ident, $_:ident) => {
    Some($p.clone())
  };
}

/// Shared template — generates 4 functions for one variant (point or CF).
///
/// Parameters:
///   $find / $into / $over / $upd  — function names
///   $param : $ptype                — parameter name and type
///   $dfn                           — distance method  (sq_dist | sq_dist_cf)
///   $ufn                           — update method    (add | add_cf)
macro_rules! __insert_variant {
  (
        $find:ident, $into:ident, $over:ident, $upd:ident,
        $param:ident : $ptype:ty,
        $dfn:ident, $ufn:ident
    ) => {
    #[inline(always)]
    pub(crate) fn $find<
      F: Float,
      CF: ClusterFeature<F>,
      D: CFDistance<F, CF>,
      A: CFDistance<F, CF>,
    >(
      tree: &mut CFTree<F, CF, D, A>,
      $param: $ptype,
    ) -> (Vec<usize>, usize) {
      let mut path: Vec<usize> = Vec::with_capacity(8);
      let mut current = tree.root;
      loop {
        path.push(current);
        if !tree.nodes[current].is_leaf() {
          let child_ids = tree.nodes[current].children();
          let mut best_id = child_ids[0];
          let mut best_dist = {
            let __cf = tree.nodes[best_id].as_cf();
            tree.dist_function.$dfn(__cf, $param, tree.dim)
          };
          for &id in &child_ids[1..] {
            let d = {
              let __cf = tree.nodes[id].as_cf();
              tree.dist_function.$dfn(__cf, $param, tree.dim)
            };
            if d < best_dist {
              best_dist = d;
              best_id = id;
            }
          }
          current = best_id;
        } else {
          break (path, current);
        }
      }
    }

    #[inline(always)]
    pub(crate) fn $into<
      F: Float,
      CF: ClusterFeature<F>,
      D: CFDistance<F, CF>,
      A: CFDistance<F, CF>,
    >(
      tree: &mut CFTree<F, CF, D, A>,
      leaf_idx: usize,
      $param: $ptype,
    ) -> Option<CF> {
      let leaf_node = &mut tree.nodes[leaf_idx];
      if leaf_node.children().is_empty() {
        let mut new_leaf_entry = CF::new(tree.dim);
        {
          let __target = &mut new_leaf_entry;
          __target.$ufn($param);
        }
        leaf_node.add_child(tree.leaf_entries.len());
        tree.leaf_entries.push(new_leaf_entry);
        return None;
      }

      let child_ids = leaf_node.children();
      let mut best_id = child_ids[0];
      let mut best_dist = {
        let __cf = &tree.leaf_entries[best_id];
        tree.dist_function.$dfn(__cf, $param, tree.dim)
      };
      for &id in &child_ids[1..] {
        let d = {
          let __cf = &tree.leaf_entries[id];
          tree.dist_function.$dfn(__cf, $param, tree.dim)
        };
        if d < best_dist {
          best_dist = d;
          best_id = id;
        }
      }

      let closest = &mut tree.leaf_entries[best_id];
      if {
        let __cf = &*closest;
        tree.abs_function.$dfn(__cf, $param, tree.dim)
      } <= tree.threshold
      {
        {
          let __target = closest;
          __target.$ufn($param);
        }
        return None;
      }

      let mut new_leaf_entry = CF::new(tree.dim);
      {
        let __target = &mut new_leaf_entry;
        __target.$ufn($param);
      }
      if leaf_node.children().len() < tree.capacity {
        leaf_node.add_child(tree.leaf_entries.len());
        tree.leaf_entries.push(new_leaf_entry);
        None
      } else {
        __overflow_ret!($param, $param, new_leaf_entry)
      }
    }

    #[inline(always)]
    pub(crate) fn $over<
      F: Float,
      CF: ClusterFeature<F>,
      D: CFDistance<F, CF>,
      A: CFDistance<F, CF>,
    >(
      tree: &mut CFTree<F, CF, D, A>,
      new_cf: CF,
      path: &mut Vec<usize>,
      $param: $ptype,
    ) {
      let mut parent_id = path.pop();
      let mut new_node_option: Option<CFNode<F, CF>> = match parent_id {
        Some(id) => {
          if tree.nodes[id].num_childs() < tree.capacity {
            let new_cf_id = tree.leaf_entries.len();
            tree.leaf_entries.push(new_cf);
            let parent = &mut tree.nodes[id];
            parent.as_mut_cf().add_cf(&tree.leaf_entries[new_cf_id]);
            parent.add_child(new_cf_id);
            None
          } else {
            let new_node_id = tree.leaf_entries.len();
            tree.leaf_entries.push(new_cf);
            let mut childs: Vec<usize> = tree.nodes[id].children().to_vec();
            childs.push(new_node_id);
            let (node1, node2) = tree.split_node(&childs, true);
            tree.nodes[id] = node1;
            Some(node2)
          }
        }
        None => unreachable!("path should never be empty during node overflow handling"),
      };

      while let Some(new_node) = new_node_option {
        parent_id = path.pop();
        match parent_id {
          Some(id) => {
            if tree.nodes[id].num_childs() < tree.capacity {
              let new_node_id = tree.nodes.len();
              tree.nodes.push(new_node);
              let parent = &mut tree.nodes[id];
              {
                let __target = parent.as_mut_cf();
                __target.$ufn($param);
              }
              parent.add_child(new_node_id);
              break;
            } else {
              let new_node_id = tree.nodes.len();
              tree.nodes.push(new_node);
              let mut childs: Vec<usize> = tree.nodes[id].children().to_vec();
              childs.push(new_node_id);
              let (node1, node2) = tree.split_node(&childs, false);
              tree.nodes[id] = node1;
              new_node_option = Some(node2);
            }
          }
          None => {
            let mut new_root: CFNode<F, CF> = CFNode::new(tree.dim, tree.capacity, false);
            new_root.add_child(tree.root);
            new_root.add_child(tree.nodes.len());
            new_root
              .as_mut_cf()
              .add_cf(tree.nodes.get(tree.root).unwrap().as_cf());
            new_root.as_mut_cf().add_cf(new_node.as_cf());
            tree.nodes.push(new_node);
            tree.root = tree.nodes.len();
            tree.nodes.push(new_root);
            break;
          }
        }
      }
    }

    #[inline(always)]
    pub(crate) fn $upd<
      F: Float,
      CF: ClusterFeature<F>,
      D: CFDistance<F, CF>,
      A: CFDistance<F, CF>,
    >(
      tree: &mut CFTree<F, CF, D, A>,
      path: &[usize],
      $param: $ptype,
    ) {
      for &index in path {
        let __target = tree.nodes[index].as_mut_cf();
        __target.$ufn($param);
      }
    }
  };
}

/// Generate all 8 insert functions from one invocation.
macro_rules! define_insert_pair {
  (
        $fp:ident / $fc:ident,
        $lp:ident / $lc:ident,
        $op:ident / $oc:ident,
        $up:ident / $uc:ident,
    ) => {
    __insert_variant! { $fp, $lp, $op, $up, point : &[F], sq_dist,  add    }
    __insert_variant! { $fc, $lc, $oc, $uc, cf    : &CF,  sq_dist_cf, add_cf }
  };
}

define_insert_pair! {
    find_leaf_path / find_leaf_path_cf,
    insert_into_leaf / insert_into_leaf_cf,
    handle_overflow_point / handle_overflow_cf,
    update_path / update_path_cf,
}
