import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tree.node import LeafNode, DecisionNode

LEAF_COLORS = {
    0: "#f8d7da",
    1: "#ffe5d0",
    2: "#fff3cd",
    3: "#e2f0d9",
    4: "#d1ecf1",
    5: "#cce5ff",
}
INTERNAL_NODE_COLOR = "#f0f4f8"


class TreeVisualizer:
    def __init__(self, dpi=200):
        self.dpi = dpi

    def save_tree_png(self, tree, output_path, user_id=None):
        if tree is None or tree.root is None:
            return

        total_nodes, total_leaves = tree.root.count_nodes_and_leaves()
        max_depth = tree.root.get_depth()

        fig_width = max(15.0, total_leaves * 2.5)
        fig_height = max(8.5, (max_depth + 1) * 2.2)

        positions = {}
        y_step = 0.82 / max(1, max_depth)
        self._assign_positions(
            tree.root, x_min=0.04, x_max=0.96, y=0.90, y_step=y_step, positions=positions
        )

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.dpi)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_axis_off()

        if user_id is not None:
            ax.set_title(
                f"Induced Binary Decision Tree — User ID = {user_id} "
                f"(Nodes: {total_nodes}, Leaves: {total_leaves}, Depth: {max_depth})",
                fontsize=13.5,
                fontweight="bold",
                pad=18,
            )

        self._draw_subtree(ax, tree.root, positions)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=LEAF_COLORS[c], ec="#2f6f3e", label=f"Rating {c}")
            for c in range(6)
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=6,
            frameon=True,
            fontsize=9,
            title="Predicted Movie Evaluation in Leaf Nodes (0–5):",
            title_fontsize=9.5,
        )

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)

    def _count_leaves(self, node):
        if node is None:
            return 0
        if isinstance(node, LeafNode):
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _assign_positions(self, node, x_min, x_max, y, y_step, positions):
        if node is None:
            return
        if isinstance(node, LeafNode):
            positions[id(node)] = (0.5 * (x_min + x_max), y)
            return

        left_leaves = max(1, self._count_leaves(node.left))
        right_leaves = max(1, self._count_leaves(node.right))
        total_leaves = left_leaves + right_leaves

        split_x = x_min + (x_max - x_min) * (left_leaves / float(total_leaves))
        positions[id(node)] = (split_x, y)

        self._assign_positions(node.left, x_min, split_x, y - y_step, y_step, positions)
        self._assign_positions(node.right, split_x, x_max, y - y_step, y_step, positions)

    def _draw_subtree(self, ax, node, positions):
        x_p, y_p = positions[id(node)]
        counts_str = "[" + ",".join(str(int(c)) for c in node.class_counts) + "]"

        if isinstance(node, DecisionNode):
            for child, is_left in [(node.left, True), (node.right, False)]:
                if child is None:
                    continue
                x_c, y_c = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(x_c, y_c + 0.03),
                    xytext=(x_p, y_p - 0.03),
                    arrowprops=dict(arrowstyle="->", color="#495057", lw=1.5),
                    zorder=1,
                )
                mid_x, mid_y = 0.5 * (x_p + x_c), 0.5 * (y_p + y_c)
                edge_lbl = "True" if is_left else "False"
                edge_col = "#1b7837" if is_left else "#b2182b"
                ax.text(
                    mid_x,
                    mid_y + 0.01,
                    edge_lbl,
                    fontsize=8.5,
                    fontweight="bold",
                    color=edge_col,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    zorder=2,
                )
                self._draw_subtree(ax, child, positions)

            label = (
                f"{node.condition.to_label()}\n"
                f"Gini: {node.impurity:.3f} | n={node.samples_count}\n"
                f"counts={counts_str}"
            )
            box = dict(boxstyle="round,pad=0.45", fc=INTERNAL_NODE_COLOR, ec="#2b5c8f", lw=1.5)
            ax.text(x_p, y_p, label, ha="center", va="center", fontsize=8, fontfamily="monospace", bbox=box, zorder=3)
        else:
            label = (
                f"Predict: {node.value}\n"
                f"Gini: {node.impurity:.3f} | n={node.samples_count}\n"
                f"counts={counts_str}"
            )
            fc_col = LEAF_COLORS.get(node.value, "#c8e6c9")
            box = dict(boxstyle="round,pad=0.45", fc=fc_col, ec="#2f6f3e", lw=1.5)
            ax.text(x_p, y_p, label, ha="center", va="center", fontsize=8, fontfamily="monospace", bbox=box, zorder=3)
