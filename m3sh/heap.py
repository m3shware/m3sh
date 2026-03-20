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

This is an alternative to Pythons :mod:`heapq` module. See [1]_ Chapter
9.3 for the array (list) based heap implementation used in this module.

Data model
----------

A tree is **heap-ordered** if the key assigned to each node is smaller
than or equal to the keys assigned to its children. A **heap** is a list
of objects with keys arranged in a complete heap-ordered binary tree,
i.e., ``key[k] <= key[2k+1]`` and ``key[k] <= key[2k+2]``.

Notes
-----
Since Python 3.14 the :mod:`heapq` module provides dedicated methods to
create max heaps.

References
----------
.. [1] Robert Sedgewick: **Algorithms in C**, *Parts 1--4*. Addison-Wesley
       Professional, 1990.
"""

class Heap:
    """ Heap data structure.

    A heap stores `(obj, key)` tuples, called heap items. When no keys are
    provided heap items reduce to one element `(object, )` tuples. In this
    case data objects act as keys.

    Heaps use the < operator for key comparison. This results in so called
    min heaps. Custom classes used as keys have to implement :meth:`__lt__`.

    Parameters
    ----------
    items : iterable, optional
        Heap items. A sequence of `(obj, key)` tuples. To save storage
        consider using :meth:`~Heap.from_list` if `items` is a list.

    Warnings
    --------
    The :mod:`heapq` modules uses heap items of the form `(key, obj)`.

    See Also
    --------
    :meth:`~Heap.from_list`, :meth:`~Heap.push`

    Notes
    -----
    Given two iterables `objs` and `keys`, one holding objects the other
    holding keys, a heap can be initialized using :func:`zip`:

    >>> heap = Heap(zip(objs, keys))
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

    # def __init__(self, items=None, *, less=(lambda x, y: x < y)):
    #     def lt(i, j):
    #         return less(self._heap[i][-1], self._heap[j][-1])

    #     # Assigning lt as instance attribute does not make it an instance
    #     # method. It does not implicitly receive a self argument, lt knowns
    #     # less and self because of closure.
    #     self._less = lt

    #     self._heap = list()
    #     self._hpos = dict()

    #     for item in items or []:
    #         self._push(item)

    def __bool__(self):
        """ Empty heap check.

        Returns
        -------
        bool
            ``True`` if the heap is not empty, ``False`` otherwise.
        """
        return len(self._heap) > 0

    def __contains__(self, obj):
        """ Containment check.

        Check if there is a heap item with ``item[0] == obj``.

        Parameters
        ----------
        obj : object
            Data object.

        Returns
        -------
        bool
            ``True`` if an item with data `obj` exists, ``False`` otherwise.
        """
        return obj in self._hpos

    def __delitem__(self, obj):
        """ Delete item.

        Delete the heap item with ``item[0] == obj``, maintains the
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
        """ Heap-ordered item iterator.

        Does not modify the heap. Heap modification during traversal has
        no effect on the traversal order.

        Returns
        -------
        iterator
            Heap item iterator.
        """
        heap = self._heap.copy()

        while heap:
            yield pop(heap)

    @classmethod
    def from_list(cls, items):
        """ Initialize heap from list.

        Turns `items` into a heap-ordered binary tree and uses it directly
        as internal storage of a heap.

        Parameters
        ----------
        items : list
            A list of `(obj, key)` tuples.

        Returns
        -------
        heap : Heap
            Heap that uses `items` directly as storage. Subsequent external
            modification of `items` will most likely destroy the heap.

        Notes
        -----
        The heap generated via ``Heap.from_list(items)`` does not duplicate
        data storage. It may be preferable to ``Heap(items)`` in some use
        cases.
        """
        heap = cls()

        heap._heap = heapify(items)
        heap._hpos = {item[0]: idx for idx, item in enumerate(heap._heap)}

        return heap

    @property
    def top(self):
        """ Access top item.

        The top item of a heap is defined to be the item with smallest key
        with respect to the < operator. Inspection of the top item does not
        change the heap. To remove the top item use :meth:`~Heap.pop`.

        Raises
        ------
        IndexError
            When trying to access the top item of an empty heap.

        See Also
        --------
        :meth:`~Heap.pop`

        Notes
        -----
        The data object of a heap item is accessible as ``item[0]`` or
        via tuple unpacking:

        >>> obj, key = heap.top
        """
        return self._heap[0]

    def pop(self):
        """ Remove top item.

        Remove top item and maintain the heap property.

        Returns
        -------
        item : tuple
            An `(obj, key)` tuple.

        Raises
        ------
        IndexError
            When trying to remove items from an empty heap.

        See Also
        --------
        top
        """
        return self._remove(0)

    def push(self, obj, key):
        """ Add item.

        Add data object with associated key. Re-adding an object replaces
        the corresponding key instead of adding a duplicate with different
        key. Maintains the heap property.

        Parameters
        ----------
        obj : object
            Data object. Has to be :term:`hashable`.
        key : object
            Key object. Has to implement :meth:`__lt__`.

        Raises
        ------
        TypeError
            If `obj` is not derived from a hashable data type.

        Notes
        -----
        The `(obj, key)` 2-tuple is called a heap item. Keys are compared
        using the < operator. See the comment on sort stability in the
        :mod:`heapq` documentation (add a counter as tie-breaker):

        >>> heap.push(obj, (key, cnt))
        """
        self._push((obj, key))

    def _push(self, item):
        """ Low-level push.

        Parameters
        ----------
        item : tuple
            A heap item, i.e., (obj, key) tuple.
        """
        # The key part may be omitted, i.e., (obj, ) tuples are fine. In this
        # case obj is used as key. This implies that obj defines __lt__ to
        # enable comparisons using the < operator.
        obj = item[0]

        try:
            # Key update is implied if obj is already stored in the heap.
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
        heap = self._heap
        n = len(heap) - 1                   # the largest valid item index

        # Swap the item at position k with its predecessor as long as it's
        # of lower priority.
        # while 0 < k <= n and self._less(k, j := (k+1)//2 - 1):
        while 0 < k <= n and heap[k][-1] < heap[j := (k+1)//2 - 1][-1]:
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
        heap = self._heap
        n = len(heap) - 1                   # the largest valid item index

        # Swap the item at position k with its successor of lower priority
        # as long as its own priority is higher. Successors of item with
        # index k have index 2k and 2k+1 (resp. 0-based 2*k+1 and 2*k+2).
        while 0 <= (j := 2*k + 1) <= n:
            # There are two successors if j < n. Get index j of child with
            # lower priority
            # if j < n and self._less(j+1, j):
            if j < n and heap[j+1][-1] < heap[j][-1]:
                j += 1

            # Swap with child of lower priority. If no swap is indicated
            # the heap property is restored.
            # if self._less(j, k):
            if heap[j][-1] < heap[k][-1]:
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
        del self._hpos[item[0]]

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
            Indices of heap items.

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

        self._hpos[self._heap[i][0]] = i
        self._hpos[self._heap[j][0]] = j


def heapify(items):
    """ Make heap-ordered binary tree.

    Shuffle elements of a list to obtain a heap-ordered binary tree.

    Parameters
    ----------
    items : list
        Heap items. A list of `(obj, key)` tuples.

    Returns
    -------
    heap : list
        The input list in heap-order.

    Warnings
    --------
    The :mod:`heapq` module function of the same name expects items of
    the form `(key, obj)`.
    """
    for i in range(1, len(items)):
        _fixup(items, i, i)

    return items


def push(heap, item):
    """ Add item.

    Add `item` to the heap-ordered list `heap`.

    Parameters
    ----------
    heap : list
        Heap-ordered list.
    item : tuple
        Heap item. An `(obj, key)` tuple

    Returns
    -------
    heap : list
        The augmented list `heap`, not a copy.

    Warnings
    --------
    Does not support key updates! Use the :class:`Heap` class to support
    key updates.
    """
    heap.append(item)
    _fixup(heap, len(heap) - 1, len(heap) - 1)

    return heap


def pop(heap):
    """ Remove top item.

    Parameters
    ----------
    heap : list
        Heap-ordered list of `(obj, key)` tuples.

    Returns
    -------
    item : tuple
        Top heap item.
    """
    _swap(heap, 0, n := len(heap) - 1)
    _fixdown(heap, 0, n - 1)

    return heap.pop()


def _check(items):
    """ Check heap property.

    Parameters
    ----------
    items : list
        Heap items.

    Returns
    -------
    bool
        ``True`` if `items` satisfies the heap property, ``False`` otherwise.
    """
    for i in range(len(items)):
        if 2*i + 1 < len(items) and items[i][-1] > items[2*i + 1][-1]:
            return False

        if 2*i + 2 < len(items) and items[i][-1] > items[2*i + 2][-1]:
            return False

    return True


def _fixup(heap, k, n):
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

    Notes
    -----
    The argument `n` is not the size of `heap`. It is the index
    ``len(heap) - 1`` of its last element (or a smaller value).
    """
    # Swap the item at position k with its predecessor at position j as
    # long as heap[j] > heap[k].
    while 0 < k <= n and heap[k][-1] < heap[j := (k+1)//2 - 1][-1]:
        _swap(heap, k, k := j)


def _fixdown(heap, k, n):
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

    Notes
    -----
    The argument `n` is not the size of `heap`. It is the index
    ``len(heap) - 1`` of its last element (or a smaller value).
    """
    # Successors of item with index k have index 2*k+1 and 2*k+2. Let j
    # denote the index of child with smaller key. Swap item at index k
    # with child at position j if heap[k] > heap[j].
    while 0 <= (j := 2*k + 1) <= n:
        # There are two successors if j < n. Get index j of child with
        # smaller key.
        if j < n and heap[j+1][-1] < heap[j][-1]:
            j += 1

        # Swap with child with smaller key. If no swap is indicated the
        # heap property is restored.
        if heap[j][-1] < heap[k][-1]:
            _swap(heap, k, k := j)
        else:
            break


def _swap(heap, i, j):
    """ Swap position of heap items.

    Parameters
    ----------
    heap : list
        Heap items.
    i, j : int
        Indices of heap items.

    Note
    ----
    Does not perform range checks on arguments `i` and `j`.
    """
    heap[i], heap[j] = heap[j], heap[i]
