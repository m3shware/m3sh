# Copyright 2024, m3shware
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Heap based priority queue implementation.

See **Algorithms in C**, *Parts 1--4* by Robert Sedgewick for the array based
heap implementation used in this module. An alternative implementation is
provided in the :py:mod:`heapq` module.
"""

class _Heap:
    """ Abstract heap base class.
    """

    def __init__(self):
        """ Initialize heap attributes.
        """
        # Internally the priority queue is modelled as a binary tree that is
        # stored in an array. _heap[0] is never used, it's only there to
        # simplify modulo computations when determining parent/child index.
        self._heap = [None]
        self._hpos = dict()

    def __bool__(self):
        """ Implicit empty heap check.
        """
        return len(self._heap) > 1

    def __contains__(self, item):
        """ Containment check.
        """
        return item in self._hpos

    def __len__(self):
        """ Return number of queued items.
        """
        return len(self._heap)-1

    def __iter__(self):
        """ Item iterator.

        Iterates over all items in the priority queue in the order items
        are stored in the underlying array object.
        """
        return iter(self._heap[1:])

    @property
    def top(self):
        """ Access item of highest priority.

        :type: 2-tuple of `object` and `priority`.

        Raises
        ------
        IndexError
            When trying to access the top element of an empty queue.
        """
        return self._heap[1]

    def pop(self):
        """ Remove item of highest priority.

        Returns
        -------
        data : object
            Object of highest priority.
        priority : float
            Object priority.

        Raises
        ------
        IndexError
            When trying to remove items from an empty queue.
        """
        # Cannot pop an empty heap
        if len(self._heap) == 1: raise IndexError

        # Remove the top element by swapping it to the end. Then shorten
        # the list and dictionary and restore the heap property.
        self._swap(1, len(self._heap)-1)
        top = self._heap.pop()
        del self._hpos[top[0]]
        self._fixdown(1)

        # Return the top element, i.e., the (item, priority) tuple of
        # highest priority.
        return top

    def push(self, data, priority):
        """ Add data object.

        Re-adding an already queued object will update the queued object's
        priority instead of adding a duplicate with a different priority,
        see :meth:`update`.

        Parameters
        ----------
        data : object
            Object to be added to the priority queue.
        priority : float
            Priority of the data object.

        Raises
        ------
        TypeError
            If `data` is not derived from a hashable data type.

        Note
        ----
        Only `hashable <https://docs.python.org/3/glossary.html#term-hashable>`_
        objects can be added to a heap. All user defined types are hashable.
        """
        if data in self._hpos:
            self.update(data, priority)
        else:
            self._hpos[data] = len(self._heap)
            self._heap.append((data, priority))
            self._fixup(len(self._heap)-1)

    def update(self, data, priority):
        """ Update priority of a data object.

        Parameters
        ----------
        data : object
            Object whose priority should be updated.
        priority : float
            New priority value.

        Raises
        ------
        KeyError
            If `data` is not an element of the queue.
        """
        # This can raise a KeyError
        k = self._hpos[data]

        # Set new priority
        self._heap[k] = (self._heap[k][0], priority)

        # Restore heap property.
        self._fixup(k)
        self._fixdown(k)

    def remove(self, data):
        """ Remove data object from heap.

        Parameters
        ----------
        data : object
            The data object to be removed.

        Raises
        ------
        KeyError
            If `data` is not queued.

        Returns
        -------
        float
            Priority of the removed object.
        """
        # This can raise a KeyError
        k = self._hpos[data]

        # Swap item at position k with last item in the heap. Nothing
        # happens if k refers to the last item in the heap ordered list.
        self._swap(k, len(self._heap)-1)

        # Remove the item (now at the last position) from the internal
        # list and delete its item from the position dictionary.
        removed_element = self._heap.pop()
        del self._hpos[removed_element[0]]

        # After shortening heap, check if k is still in the valid range
        # of indices, ranging from 1 up to and including len(heap)-1.
        # Always true unless we removed the final element of the heap.
        if k < len(self._heap):
            self._fixup(k)
            self._fixdown(k)

        # The priority of the removed object.
        return removed_element[1]

    def _swap(self, i, j):
        """ Swap position of heap items.

        Swapping two heap elements will invalidate the heap property. This has
        to be fixed by subsequent calls to :meth:`_fixup` and :meth:`_fixdown`.

        Parameters
        ----------
        i : int
            Position of a heap element
        j : int
            Position of a heap element

        Raises
        ------
        IndexError
            If any of the given indices are out of bounds.
        """
        # Update position information in dictionary
        self._hpos[self._heap[i][0]] = j
        self._hpos[self._heap[j][0]] = i

        # Swap items without temporary storage
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]


class MaxHeap(_Heap):
    """ Priority queue.

    Larger priority values signify higher priority. The :attr:`top` element of
    a priority queue is the object of highest priority.

    Parameters
    ----------
    items : iterable, optional
        A sequence of `(object, priority)` pairs. Generically, `priority` is
        a numeric data type.

    Raises
    ------
    TypeError
        If `items` is not iterable.

    Note
    ----
    Only `hashable <https://docs.python.org/3/glossary.html#term-hashable>`_
    objects can be added to a heap. All user defined types are hashable.
    """

    def __init__(self, items=None):
        """ Initialize heap from iterable.
        """
        super().__init__()

        if items is not None:
            for item in items:
                self.push(*item)

    def _fixup(self, k):
        """ Restore heap property.

        Fixes the heap property upwards starting from the item at the
        given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # Swap the item at position k with its predecessor as long as it's
        # of higher priority.
        while k > 1 and self._heap[k//2][1] < self._heap[k][1]:
            self._swap(k, k//2)
            k = k//2

    def _fixdown(self, k):
        """ Restore heap property.

        Fixes the heap property downwards starting from the item at
        the given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # The largest index of any valid heap item.
        n = len(self._heap)-1

        # Swap the item at position k with its successor of higher priority
        # as long as its own priority is lower. Successors of item with
        # index k have index 2k and 2k+1.
        while 2*k <= n:
            # Index of the left child of item with index k.
            j = 2*k

            # There are two successors if j < n. Get index j of child with
            # higher priority
            if j < n and self._heap[j][1] < self._heap[j+1][1]:
                j += 1

            # If no swap is indicated the heap property is restored.
            if self._heap[k][1] >= self._heap[j][1]:
                break

            # Swap with child of higher priority.
            self._swap(j, k)
            k = j


class MinHeap(_Heap):
    """ Priority queue.

    Smaller priority values signify higher priority. The :attr:`top` element of
    a priority queue is the object of highest priority.

    Parameters
    ----------
    items : iterable, optional
        A sequence of `(object, priority)` pairs. Generically, `priority` is
        a numeric data type.

    Raises
    ------
    TypeError
        If `items` is not iterable.

    Note
    ----
    Only `hashable <https://docs.python.org/3/glossary.html#term-hashable>`_
    objects can be added to a heap. All user defined types are hashable.
    """

    def __init__(self, items=None):
        """ Initialize heap from iterable.
        """
        super().__init__()

        if items is not None:
            for item in items:
                self.push(*item)

    def _fixup(self, k):
        """ Restore heap property.

        Fixes the heap property upwards starting from the item at the
        given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # Swap the item at position k with its predecessor as long as it's
        # of lower priority.
        while k > 1 and self._heap[k//2][1] > self._heap[k][1]:
            self._swap(k, k//2)
            k = k//2

    def _fixdown(self, k):
        """ Restore heap property.

        Fixes the heap property downwards starting from the item at
        the given position.

        Parameters
        ----------
        k : int
            Index of heap element that violates the heap property.
        """
        # The largest index of any valid heap item.
        n = len(self._heap)-1

        # Swap the item at position k with its successor of lower priority
        # as long as its own priority is higher. Successors of item with
        # index k have index 2k and 2k+1.
        while 2*k <= n:
            # Index of the left child of item with index k.
            j = 2*k

            # There are two successors if j < n. Get index j of child with
            # lower priority
            if j < n and self._heap[j][1] > self._heap[j+1][1]:
                j += 1

            # If no swap is indicated the heap property is restored.
            if self._heap[k][1] <= self._heap[j][1]:
                break

            # Swap with child of lower priority, if any
            self._swap(j, k)
            k = j