import numpy as np
import scipy.cluster.hierarchy as hierarchy


_link_line_colors_default = ['lightcoral','steelblue','darkseagreen', 'mediumorchid', 'gold', 'deepskyblue']
_link_line_colors = list(_link_line_colors_default)

# adapted from scipy.cluster.hierarchy.dendrogram
def dendrogram(Z, p=30, truncate_mode=None, color_threshold=None, truncate_threshold=0.5, min_clus_size = None,
               get_leaves=True, orientation='top', labels=None,
               count_sort=False, distance_sort=False, show_leaf_counts=True,
               no_plot=False, no_labels=False, leaf_font_size=None,
               leaf_rotation=None, leaf_label_func=None,
               show_contracted=False, link_color_func=None, ax=None,
               above_threshold_color='C0', hasDuplicates = True):
    """
    his is ane extended version of scipy.cluster.hierarchy.dendrogram.
    For detailed explanations see scipy.cluster.hierarchy.dendrogram. 
    Additional functionality: this version offers the additional truncation_mode: 'threshold' which truncates a dendrogram for distances
        below thae specified truncate_threshold. This mode offers the additional possibilities of specifiying the maximum cluster size through p and 
        the min_clus_size which removes clusters below the specified sice from visualisation.
        For a truncate_threshold of 1, this function performs a truncation based on maximum cluster size alone.
    
    Additional Parameters or 
    ----------
    
    p : int, optional
        The ``p`` parameter for ``truncate_mode``. Here used for specifying a maximum cluster size.
    truncate_mode : str, optional
        'threshold': clusters are formed based on distance below certain value specified by ``truncate_theshold``
    truncate_threshold: double, optional
        parameter for truncate_mode 'threshold'
    min_clus_size: int, optional
        the minimum number of leaves within a cluster that are necessary for the cluster to be added to the dendrogram
    hasDuplicates: bool
        if True, will be ensured that dendrogram will be computed on dataset cleaned of duplicates with counts being adjusted later on,
            (the leaf label function has been adapted to allow for a count readjustment)
    
    Returns
    -------
    R : dict
        A dictionary of data structures computed to render the
        dendrogram, additional keys added: 
          Its has the following keys:
        ``'truncated_leaves'``
          A list of tuples of the form (cluster_number, label), indicating to which cluster the label belongs
    """
    
    Z = np.asarray(Z, order='c')

    if orientation not in ["top", "left", "bottom", "right"]:
        raise ValueError("orientation must be one of 'top', 'left', "
                         "'bottom', or 'right'")
    
    duplicate_labels = None
    if hasDuplicates == True: 
        duplicate_labels = labels
        labels = list(set(labels))
    
    if labels is not None and Z.shape[0] + 1 != len(labels):
        raise ValueError("Dimensions of Z and labels must be consistent.")

    hierarchy.is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1
    if type(p) in (int, float):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')

    if truncate_mode not in ('lastp', 'mtica', 'level', 'threshold', 'none', None):
        # 'mtica' is kept working for backwards compat.
        raise ValueError('Invalid truncation mode.')
    
    if truncate_mode != 'threshold' and min_clus_size is not None and min_clus_size > 1:
         raise ValueError('Can only set min_clus_size for truncation_mode: threshold')

    if truncate_mode == 'lastp':
        if p > n or p == 0:
            p = n

    if truncate_mode == 'mtica':
        # 'mtica' is an alias
        truncate_mode = 'level'
    
    # new with threshold
    if truncate_mode == 'level' or truncate_mode == 'threshold':
        if p <= 0:
            p = np.inf 
    if type(min_clus_size) is not int and min_clus_size is not None:
        raise TypeError('The min_clus_size must be a number or None')
    if min_clus_size is not None and min_clus_size <= 0:
        min_clus_size = None

    if get_leaves:
        lvs = []
    else:
        lvs = None

    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []  # list of leaves

    if color_threshold is None or (isinstance(color_threshold, str) and
                                   color_threshold == 'default'):
        color_threshold = max(Z[:, 2]) * 0.7

    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl,
         'leaves': lvs, 'color_list': color_list}

    # Empty list will be filled in _dendrogram_calculate_info
    contraction_marks = [] if show_contracted else None
    # NEW : to be filled with groups of truncated clusters
    truncated_leaves = []
    
    _dendrogram_calculate_info(
        Z=Z, p=p,
        truncate_mode=truncate_mode,
        truncate_threshold = truncate_threshold, 
        color_threshold=color_threshold,
        min_clus_size=min_clus_size,
        get_leaves=get_leaves,
        orientation=orientation,
        labels=labels,
        duplicate_labels = duplicate_labels,
        count_sort=count_sort,
        distance_sort=distance_sort,
        show_leaf_counts=show_leaf_counts,
        i=2*n - 2,
        iv=0.0,
        ivl=ivl,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        lvs=lvs,
        current_color=current_color,
        color_list=color_list,
        currently_below_threshold=currently_below_threshold,
        leaf_label_func=leaf_label_func,
        contraction_marks=contraction_marks,
        truncated_leaves=truncated_leaves,
        link_color_func=link_color_func,
        above_threshold_color=above_threshold_color)

    if not no_plot:
        mh = max(Z[:, 2])
        hierarchy._plot_dendrogram(icoord_list, dcoord_list, ivl, p, n, mh, orientation,
                         no_labels, color_list,
                         leaf_font_size=leaf_font_size,
                         leaf_rotation=leaf_rotation,
                         contraction_marks=contraction_marks,
                         ax=ax,
                         above_threshold_color=above_threshold_color)

    R["leaves_color_list"] = hierarchy._get_leaves_color_list(R)
    #Newly added list of truncated leaf groups to R
    R["truncated_leaves"] = truncated_leaves

    return R


def _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                i, labels, truncated_leaves):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.
    if truncated_leaves is not None and labels is not None: 
        _get_truncated_leaves(Z, i, i, n, truncated_leaves, labels)
    if lvs is not None:
        lvs.append(int(i))

    # If leaf node labels are to be displayed...
    if ivl is not None:
        # If a leaf_label_func has been provided, the label comes from the
        # string returned from the leaf_label_func, which is a function
        # passed to dendrogram.
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i), str(1)))
        else:
            # Otherwise, if the dendrogram caller has passed a labels list
            # for the leaf nodes, use it.
            if labels is not None:
                ivl.append(labels[int(i - n)])
            else:
                # Otherwise, use the id as the label for the leaf.x
                ivl.append(str(int(i)))


def _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                   i, labels, show_leaf_counts, truncated_leaves):
    # If the leaf id structure is not None and is a list then the caller
    # to dendrogram has indicated that cluster id's corresponding to the
    # leaf nodes should be recorded.
    if truncated_leaves is not None and labels is not None: 
        _get_truncated_leaves(Z, i, i, n, truncated_leaves, labels)
    if lvs is not None:
        lvs.append(int(i))
    if ivl is not None:
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i), str(int(Z[i - n, 3]))))
        else:
            if show_leaf_counts:
                ivl.append("(" + str(int(Z[i - n, 3])) + ")")
            else:
                ivl.append("")


def _get_truncated_leaves(Z, clus_id, i, n, truncated_leaves, labels): 
    # produces list of leaves specifying which truncated cluster they belong to
    if i < n:
        truncated_leaves.append((clus_id, labels[int(i - n)]))
    else: 
        _get_truncated_leaves(Z, clus_id, int(Z[i - n, 0]), n, truncated_leaves, labels)
        _get_truncated_leaves(Z, clus_id, int(Z[i - n, 1]), n, truncated_leaves, labels) 


def _dendrogram_calculate_info(Z, p, truncate_mode, truncate_threshold, 
                               color_threshold=np.inf, min_clus_size=None,get_leaves=True,
                               orientation='top', labels=None, duplicate_labels = None,
                               count_sort=False, distance_sort=False,
                               show_leaf_counts=False, i=-1, iv=0.0,
                               ivl=[], n=0, icoord_list=[], dcoord_list=[],
                               lvs=None, mhr=False,
                               current_color=[], color_list=[],
                               currently_below_threshold=[],
                               leaf_label_func=None, level=0,
                               contraction_marks=None,
                               truncated_leaves=None,
                               link_color_func=None,
                               above_threshold_color='C0'):
    """
    For detailed explanations see scipy.cluster.hierarchy.dendrogram. 
    Calculate the endpoints of the links as well as the labels for the
    the dendrogram, added functionality for the truncation_mode 'threshold'
    """
    
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")

    if i == -1:
        raise ValueError("Invalid root cluster index i.")

    if truncate_mode == 'lastp':
        # If the node is a leaf node but corresponds to a non-singleton
        # cluster, its label is either the empty string or the number of
        # original observations belonging to cluster i.
        if 2*n - p > i >= n:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts, truncated_leaves)
            if contraction_marks is not None:
                hierarchy._append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels, truncated_leaves)
            return (iv + 5.0, 10.0, 0.0, 0.0)
        
    elif truncate_mode == 'level':
        if i > n and level > p:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts, truncated_leaves)
            if contraction_marks is not None:
                hierarchy._append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels, truncated_leaves)
            return (iv + 5.0, 10.0, 0.0, 0.0)
        
    # added truncation mode
    elif truncate_mode == 'threshold':     
        if i > n and duplicate_labels != None: 
            Z[i - n, 3]= _get_duplicate_cluster_size(Z, i, n, labels, duplicate_labels)
        if i > n and Z[i - n, 2] < truncate_threshold and truncate_threshold > 0 and int(Z[i - n, 3]) < p: 
            d = Z[i - n, 2]
            
            if min_clus_size is not None and Z[i - n, 3] < min_clus_size:
                return (None, None, None, None)
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts, truncated_leaves)
            if contraction_marks is not None:
                hierarchy._append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            
            if min_clus_size is not None and min_clus_size > 1: 
                return (None, None, None, None)

            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                       leaf_label_func, i, labels, truncated_leaves)
            
            return (iv + 5.0, 10.0, 0.0, 0.0)
        
    # Otherwise, only truncate if we have a leaf node.
    #
    # Only place leaves if they correspond to original observations.
    if i < n:
        if min_clus_size is not None and min_clus_size > 1: 
                return (None, None, None, None)
        _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                   leaf_label_func, i, labels, truncated_leaves)
        return (iv + 5.0, 10.0, 0.0, 0.0)

    # !!! Otherwise, we don't have a leaf node, so work on plotting a
    # non-leaf node.
    # Actual indices of a and b
    aa = int(Z[i - n, 0])
    ab = int(Z[i - n, 1])
    if aa >= n:
        # The number of singletons below cluster a
        na = Z[aa - n, 3]
        # The distance between a's two direct children.
        da = Z[aa - n, 2]
    else:
        na = 1
        da = 0.0
    if ab >= n:
        nb = Z[ab - n, 3]
        db = Z[ab - n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort == True:
        # If a has a count greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if na > nb:
            # The cluster index to draw to the left (ua) will be ab
            # and the one to draw to the right (ub) will be aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # If a has a count less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort == True:
        # If a has a distance greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # If a has a distance less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        ua = aa
        ub = ab


    # Updated iv variable and the amount of space used.
    (uiva, uwa, uah, uamd) = \
        _dendrogram_calculate_info(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            truncate_threshold=truncate_threshold,
            color_threshold=color_threshold,
            min_clus_size=min_clus_size,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            duplicate_labels = duplicate_labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ua, iv=iv, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            truncated_leaves=truncated_leaves,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    
    h = Z[i - n, 2]
    if h >= color_threshold or color_threshold <= 0:
        c = above_threshold_color
        
        if currently_below_threshold[0]:
            current_color[0] = (current_color[0] + 1) % len(_link_line_colors)
        currently_below_threshold[0] = False
    else:
        currently_below_threshold[0] = True
        c = _link_line_colors[current_color[0]]
    #print(c)
    if uwa is not None: 
        (uivb, uwb, ubh, ubmd) = \
            _dendrogram_calculate_info(
                Z=Z, p=p,
                truncate_mode=truncate_mode,
                truncate_threshold=truncate_threshold,
                color_threshold=color_threshold,
                min_clus_size=min_clus_size,
                get_leaves=get_leaves,
                orientation=orientation,
                labels=labels,
                duplicate_labels = duplicate_labels,
                count_sort=count_sort,
                distance_sort=distance_sort,
                show_leaf_counts=show_leaf_counts,
                i=ub, iv=iv + uwa, ivl=ivl, n=n,
                icoord_list=icoord_list,
                dcoord_list=dcoord_list, lvs=lvs,
                current_color=current_color,
                color_list=color_list,
                currently_below_threshold=currently_below_threshold,
                leaf_label_func=leaf_label_func,
                level=level + 1, contraction_marks=contraction_marks,
                truncated_leaves=truncated_leaves, 
                link_color_func=link_color_func,
                above_threshold_color=above_threshold_color)
    else: 
        (uivb, uwb, ubh, ubmd) = \
            _dendrogram_calculate_info(
                Z=Z, p=p,
                truncate_mode=truncate_mode,
                truncate_threshold=truncate_threshold,
                color_threshold=color_threshold,
                min_clus_size=min_clus_size,
                get_leaves=get_leaves,
                orientation=orientation,
                labels=labels,
                duplicate_labels = duplicate_labels,
                count_sort=count_sort,
                distance_sort=distance_sort,
                show_leaf_counts=show_leaf_counts,
                i=ub, iv=iv, ivl=ivl, n=n,
                icoord_list=icoord_list,
                dcoord_list=dcoord_list, lvs=lvs,
                current_color=current_color,
                color_list=color_list,
                currently_below_threshold=currently_below_threshold,
                leaf_label_func=leaf_label_func,
                level=level + 1, contraction_marks=contraction_marks,
                truncated_leaves=truncated_leaves, 
                link_color_func=link_color_func,
                above_threshold_color=above_threshold_color)
    if (uivb, uwb, ubh, ubmd) != (None, None, None, None) and (uiva, uwa, uah, uamd) != (None, None, None, None): 
        max_dist = max(uamd, ubmd, h)

        
        # in case of an exclusion of cluster of smaller size, adjustment of length of links (needed due to ignored child-links)
        if min_clus_size is not None:
            x_coor = uiva + uivb / 2
            uah_adjust = 0
            ubh_adjust = 0
            has_child_left = False
            has_child_right = False
            #check if there is a child link-middle point with the same x-coordinate as the current link
            for i in range(len(icoord_list)):
                x_coor = (icoord_list[i][0] + icoord_list[i][2]) / 2
                if x_coor == uiva:
                    has_child_left = True
                    uah_adjust = uah - dcoord_list[i][1]
                if x_coor == uivb:
                    has_child_right = True
                    ubh_adjust = ubh - dcoord_list[i][1]
            #else the link goes to a leaf node
            if has_child_left is False: 
                uah = 0.0
            if has_child_right is False: 
                ubh = 0.0
            
            dcoord_list.append([uah- uah_adjust, h, h, ubh - ubh_adjust])
        else: 
            dcoord_list.append([uah, h, h, ubh])
        icoord_list.append([uiva, uiva, uivb, uivb])

        if link_color_func is not None:
            v = link_color_func(int(i))
            if not isinstance(v, str):
                raise TypeError("link_color_func must return a matplotlib "
                                "color string!")
            color_list.append(v)
        else:
            color_list.append(c)
        return (((uiva + uivb) / 2), uwa + uwb, h, max_dist)
    elif (uivb, uwb, ubh, ubmd) == (None, None, None, None) and  (uiva, uwa, uah, uamd) == (None, None, None, None): 
        return  (None, None, None, None)
    elif (uivb, uwb, ubh, ubmd) == (None, None, None, None) and  (uiva, uwa, uah, uamd) != (None, None, None, None): 
        return ((uiva), uwa, h, max(uamd, h))
    else:
        return (((uivb)), uwb, h, max(ubmd, h))
        

def _get_duplicate_cluster_size(Z, i, n, labels, duplicate_labels):
    
    # adjusts leaf counts to incorporate duplicates
    
    c_temp = 0
    t_leaves = []
    _get_truncated_leaves(Z, i, i, n, t_leaves, labels)
    for t in t_leaves: 
        c_temp += duplicate_labels.count(t[1])
    return c_temp