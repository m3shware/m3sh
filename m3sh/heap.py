# Copyright 2024-26, m3shware developers
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

""" Priority queues and heaps.

This is an alternative to Pythons :mod:`heapq` module. See [1]_ Chapter 9.3
for the array (list) based heap implementation used in this module.

References
----------
.. [1] Robert Sedgewick: **Algorithms in C**, *Parts 1--4*. Addison-Wesley
       Professional, 1990.
"""

# A tree is **heap-ordered** if the key assigned to each node is smaller
# than or equal to the keys assigned to its children. A **heap** is a list
# of objects with keys arranged in a complete heap-ordered binary tree,
# i.e., ``heap[k] <= heap[2k+1]`` and ``heap[k] <= heap[2k+2]``.

# This definition is slightly different from the one given in most textbooks
# in two important aspects: we use 0-based indexing and sort using the <
# relation.

class Heap:
    """ Heap base class.

    A heap stores `(key, object)` tuples, called heap items. When no keys
    are provided heap items reduce to one element `(object, )` tuples. In
    this case data objects act as keys.

    Parameters
    ----------
    items : iterable
        Heap items.

    See Also
    --------
    :meth:`~Heap.push`
    """

    def __init__(self, items=None):
        # Internally the priority queue is modelled as a binary tree that is
        # stored in an list. For 1-based indexing the children of node with
        # index k have index 2k and 2k+1. The parent of node k is obtained as
        # k//2 (integer division). We use 0-based indexing. In this case the
        # node with index k has children 2k+1 and 2k+2. The parent node index
        # obtained as (k+1)//2 - 1.
        self._heap = list()
        self._hpos = dict()

        for item in items or []:
            self._push(item)

    # def __init__(self, *, less=(lambda x, y: x < y)):
    #     def lt(i, j):
    #         return less(self._heap[i], self._heap[j])

    #     # Assign lt as instance attribute does not make it an instance
    #     # method. It does not implicitly received a self argument, lt
    #     # knowns less and self because of closure.
    #     self._less = lt

    def __bool__(self):
        """ Empty heap check.

        Returns
        -------
        bool
            :obj:`True` if the heap is not empty, :obj:`False` otherwise.
        """
        return len(self._heap) > 0

    def __contains__(self, obj):
        """ Containment check.

        Check if there is a heap item with ``item[-1] == obj``.

        Parameters
        ----------
        obj : object
            Data object.

        Returns
        -------
        bool
            :obj:`True` if an item with data `obj` exists, :obj:`False`
            otherwise.
        """
        return obj in self._hpos

    def __delitem__(self, obj):
        """ Delete item.

        Delete the heap item with ``item[-1] == obj``, maintains the
        heap property.

        Parameters
        ----------
        obj : object
            Data object.

        Raises
        ------
        KeyError
            If no heap item with data `obj` exists.
        """
        self._remove(self._hpos[obj])

    def __len__(self):
        """ Size of heap.

        Returns
        -------
        int
            Number of heap items.
        """
        return len(self._heap)

    def __iter__(self):
        """ Item iterator.

        Iterates over all heap items in the order items are stored in the
        underlying list.

        Returns
        -------
        iterator
            Heap item iterator.

        Warnings
        --------
        This is **not** heap-order traversal!
        """
        return iter(self._heap)

    # @classmethod
    # # def from_iterable(cls, objs, keys=None, *, less=lambda x, y: x < y):
    # def from_iterable(cls, objs, keys=None):
    #     """ Construct heap from iterable.

    #     Parameters
    #     ----------
    #     objs : iterable
    #         Data objects. All data objects have to be :term:`hashable`.
    #     keys : iterable, optional
    #         Corresponding key objects.

    #     Raises
    #     ------
    #     ValueError
    #         If the number of keys does not match the number of objects.
    #     """
    #     # heap = cls(less=less)
    #     heap = cls()

    #     if keys is None:
    #         for obj in objs:
    #             heap.push(obj)
    #     else:
    #         for obj, key in zip(objs, keys, strict=True):
    #             heap.push(obj, key)

    #     return heap

    @classmethod
    def from_iterable(cls, items):
        """ Construct heap from iterable.

        Parameters
        ----------
        items : iterable
            An iterable of `(object, key)` or `(object, )` tuples. The
            order of tuple elements matches the argument order of
            :meth:`~Heap.push`.

        Returns
        -------
        heap : Heap
            Heap instance.

        Notes
        -----
        Given two iterables `objs` and `keys`, one holding object and one
        holding keys, feasible input for this method can be produced with
        :func:`zip`:

        >>> heap = Heap.from_iterable(zip(objs, keys))
        """
        heap = cls()

        # Item tuples are not directly stored in the heap defining list.
        # Their element order is inverted to match the argument order of
        # the push method.
        for item in items:
            heap.push(*item)

        return heap

    @property
    def top(self):
        """ Access top item.

        The top item of a heap is defined to be the smallest item with
        respect to the < operator.

        Raises
        ------
        IndexError
            When trying to access the top item of an empty heap.

        See Also
        --------
        :meth:`~Heap.pop`

        Notes
        -----
        The data object of a heap item is accessible as ``item[-1]``.
        """
        return self._heap[0]

    def pop(self):
        """ Remove top item.

        Romve top item and maintain the heap property.

        Returns
        -------
        item : tuple
            Either `(key, object)` or `(object, )` tuple. The data object
            of a heap item is always accessible as ``item[-1]``.

        Raises
        ------
        IndexError
            When trying to remove items from an empty heap.
        """
        return self._remove(0)

    def push(self, obj, key=None):
        """ Add item.

        Add data object with associated key. Re-adding an object replaces
        the corresponding key instead of adding a duplicate with different
        key. Maintains the heap property.

        Parameters
        ----------
        obj : object
            Data object. Has to be :term:`hashable`.
        key : object, optional
            Key object.

        Raises
        ------
        TypeError
            If `obj` is not derived from a hashable data type.

        Notes
        -----
        The `(key, obj)` 2-tuple (or the 1-tuple `(obj, )` if no key is
        provided) is called a heap item. Heap items are compared using the
        < operator. Consequently, data objects act as tie breakers when
        keys compare equal, and objects act as keys if no keys are provided.
        """
        self._push((obj, ) if key is None else (key, obj))

        # try:
        #     k = self._hpos[obj]
        # except KeyError:
        #     # The following three lines of code could be put in a dedicated
        #     # method called _append().
        #     self._hpos[obj] = len(self._heap)
        #     self._heap.append(item)
        #     self._fixup(len(self._heap) - 1)
        # else:
        #     # The following three lines of code could be put in a dedicated
        #     # method called _update_replace().
        #     self._heap[k] = item

        #     # Depending on the key value only one of the following methods
        #     # will do some actual computation.
        #     self._fixup(k)
        #     self._fixdown(k)

    def _push(self, item):
        """ Add item.

        A low-level version of :meth:`~Heap.push`.

        Parameters
        ----------
        item : tuple
            A heap item.
        """
        obj = item[-1]

        try:
            k = self._hpos[obj]
        except KeyError:
            # The following three lines of code could be put in a dedicated
            # method called _append().
            self._hpos[obj] = len(self._heap)
            self._heap.append(item)
            self._fixup(len(self._heap) - 1)
        else:
            # The following three lines of code could be put in a dedicated
            # method called _update_replace().
            self._heap[k] = item

            # Depending on the key value only one of the following methods
            # will do some actual computation.
            self._fixup(k)
            self._fixdown(k)

    def _fixup(self, k):
        """ Restore heap property.

        Fixes the heap property upwards starting from the item at the
        given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # The largest valid index of any heap item.
        n = len(self._heap) - 1

        # Swap the item at position k with its predecessor as long as it's
        # of lower priority.
        # while 0 < k <= n and self._less(k, j := (k+1)//2 - 1):
        while 0 < k <= n and self._heap[k] < self._heap[j := (k+1)//2 - 1]:
            self._swap(k, k := j)

    def _fixdown(self, k):
        """ Restore heap property.

        Fixes the heap property downwards starting from the item at the
        given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # The largest valid index of any heap item.
        n = len(self._heap) - 1

        # Swap the item at position k with its successor of lower priority
        # as long as its own priority is higher. Successors of item with
        # index k have index 2k and 2k+1 (resp. 0-based 2*k+1 and 2*k+2).
        while 0 <= (j := 2*k + 1) <= n:
            # There are two successors if j < n. Get index j of child with
            # lower priority
            # if j < n and self._less(j+1, j):
            if j < n and self._heap[j+1] < self._heap[j]:
                j += 1

            # Swap with child of lower priority. If no swap is indicated
            # the heap property is restored.
            # if self._less(j, k):
            if self._heap[j] < self._heap[k]:
                self._swap(k, k := j)
            else:
                break

    def _remove(self, k):
        """ Remove item from heap.

        Remove heap item at a given position.

        Parameters
        ----------
        k : int
            Index of item to remove.

        Raises
        ------
        IndexError
            If `k` is out of bounds.

        Returns
        -------
        item : tuple
            Heap item that has been removed.
        """
        # Swap item at position k with last item in the heap. Nothing
        # happens if k refers to the last item in the heap ordered list.
        # For any out of bounds values _swap() raises IndexError.
        self._swap(k, len(self._heap) - 1)

        # Remove the item (now at the last position) from the internal
        # list and delete it item from the position dictionary.
        item = self._heap.pop()
        del self._hpos[item[-1]]

        # Fix the heap property after shortening heap. Nothing happens
        # if k refers to an invalid item (when removing the last item).
        self._fixup(k)
        self._fixdown(k)

        return item

    def _swap(self, i, j):
        """ Swap position of heap items.

        Parameters
        ----------
        i, j : int
            Index of heap elements.

        Raises
        ------
        IndexError
            If any of the given indices are out of bounds.
        """
        # Do not use negative indices even if they are valid! Negative
        # indices indicate an error in a calling method.
        assert i >= 0
        assert j >= 0

        # Swap items. Raises index error when position values are out of
        # bounds. Negative indices are invalid!
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

        self._hpos[self._heap[i][-1]] = i
        self._hpos[self._heap[j][-1]] = j


def check(items):
    """ Check heap property.

    Parameters
    ----------
    items : list
        Heap items.

    Returns
    -------
    bool
        :obj:`True` if `items` satisfies the heap property, :obj:`False`
        otherwise.
    """
    for i in range(len(items)):
        if 2*i + 1 < len(items) and items[i] > items[2*i + 1]:
            return False

        if 2*i + 2 < len(items) and items[i] > items[2*i + 2]:
            return False

    return True


def heapify(items):
    """ Make heap-ordered binary tree.

    Move elements of a list to obtain a heap-ordered binary tree.

    Parameters
    ----------
    items : list
        Heap items (data objects).

    Returns
    -------
    list
        The input list in heap-order.

    Note
    ----
    List elements are directly compared using the :math:`<` operator.
    """
    for i in range(1, len(items)):
        fixup(items, i, i)

    return items


def push(heap, item):
    """ Add item.

    Parameters
    ----------
    heap : list
        Heap-ordered list.
    item : object
        Heap item.
    """
    heap.append(item)
    fixup(heap, len(heap) - 1, len(heap) - 1)


def pop(heap):
    """ Remove top item.

    Parameters
    ----------
    heap : list
        Heap-ordered list.

    Returns
    -------
    object
        Top heap item.
    """
    swap(heap, 0, n := len(heap) - 1)
    fixdown(heap, 0, n - 1)

    return heap.pop()


def fixup(heap, k, n):
    """ Restore heap property upwards.

    Fix the heap property upwards starting at the item with index `k`.

    Parameters
    ----------
    heap : list
        Heap items.
    k : int
        Index of heap element that violates the heap property.
    n : int
        Index of last heap element.

    Note
    ----
    The argument `n` is not the size of `heap` but the index
    ``len(heap) - 1`` of its last element (or a smaller value).
    """
    # Swap the item at position k with its predecessor at position j as
    # long as heap[j] > heap[k].
    while 0 < k <= n and heap[k] < heap[j := (k+1)//2 - 1]:
        swap(heap, k, k := j)


def fixdown(heap, k, n):
    """ Restore heap property downwards.

    Fix the heap property downwards starting at the item with index `k`.

    Parameters
    ----------
    heap : list
        Heap items.
    k : int
        Index of heap element that violates the heap property.
    n : int
        Index of last heap element.

    Note
    ----
    The argument `n` is not the size of `heap` but the index
    ``len(heap) - 1`` of its last element (or a smaller value).
    """
    # Successors of item with index k have index 2*k+1 and 2*k+2. Let j
    # denote the index of child with smaller key. Swap item at index k
    # with child at position j if heap[k] > heap[j].
    while 0 <= (j := 2*k + 1) <= n:
        # There are two successors if j < n. Get index j of child with
        # smaller key.
        if j < n and heap[j+1] < heap[j]:
            j += 1

        # Swap with child with smaller key. If no swap is indicated the
        # heap property is restored.
        if heap[j] < heap[k]:
            swap(heap, k, k := j)
        else:
            break


def swap(heap, i, j):
    """ Swap position of heap items.

    Parameters
    ----------
    heap : list
        Heap items.
    i, j : int
        Index of heap item.

    Note
    ----
    Does not perform range checks on arguments `i` and `j`.
    """
    heap[i], heap[j] = heap[j], heap[i]
