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

""" Visualization using VTK.

Wrapper functions for `VTK <https://vtk.org/doc/nightly/html>`_ functionality.
This is not meant as a full featured set of visualization routines but should
serve as a quick and convenient way to achieve basic visualization tasks.

This module can also be used as a stand-alone OBJ viewer:

>>> python vis.py file.obj --edges

opens a graphics window and displays the contents of `file.obj`. Omitting
the '--edges' argument will not render mesh edges.
"""

import numpy as np
import vtk

from vtk.util import colors
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy


# Main render window. Always created show(). Windows created by window()
# are considered secondary windows. They are closed once the main window
# is closed.
_renwin = None

# The current render of the main window. This is either the last render
# created or set as current via canvas().
_renderer = None

# Overlay viewport of the main window. Renderer on layer 1, transparent
# background by default. All other renderers at on layer 0 and opaque.
# This render is created and removed on demand (default key 'h' for help).
_splash = None

# All renderers of the main window. Does not include renderers of secondary
# windows which are currently limited to a single renderer on layer 0.
_renderers = []

# All render window interactors including the interactors of secondary
# windows.
_interactors = []


def canvas(*args, color=None, color2=None, camera=None, transparent=None,
           interactive=None, layer=None):
    r""" Create or modify viewport.

    Change properties of an existing viewport or create a new one inside
    the main render window. A viewport's size and position is defined
    relative to the size of the render window in normalized coordinates.

    The variable length argument list `*args` may hold 0 (to create new
    viewport), 1 (to make `renderer` current), 4 (to set viewport dimensions)
    or 5 values as documented below.

    Parameters
    ----------
    renderer : vtkRenderer, optional
        Viewport identifier.
    xmin : float, optional
        Smaller x-coordinate of the viewport.
    xmax : float, optional
        Larger x-coordinate of the viewport.
    ymin : float, optional
        Smaller y-coordinate of the viewport.
    ymax : float, optional
        Larger y-coordinate of the viewport.

    Returns
    -------
    vtkRenderer
        Active viewport. All subsequent plotting happen in this viewport.

    Note
    ----
    Passing a renderer as first positional argument makes it the current
    viewport. All :obj:`None` keyword arguments do not modify the
    corresponding property of the current viewport.

    Keyword Arguments
    -----------------
    color : array_like, shape (3, ), optional
        Background color.
    color2 : array_like, shape (3, ), optional
        Top background color.
    camera : vtkRenderer or vtkCamera, optional
        Shared camera.
    transparent : bool, optional
        Toggle transparent background.
    interactive : bool, optional
        Toggle event notification.
    layer : int, optional
        Layer index. Only for internal use.

    Note
    ----
    Setting `interactive` to :obj:`False` will prevent a viewport from
    receiving events. Note that widgets placed in such a non-interactive
    renderer still receive interaction events.
    """
    # The active renderer is a global state variable that is modified
    # inside this function.
    global _renderer

    if len(args) == 1 or len(args) == 5:
        # A renderer is given. Change its properties if they are given.
        # The passed renderer also becomes the active renderer.
        _renderer = args[0]

        # Remove the viewport argument from the argument list for easy
        # access of the vieport dimension parameters.
        args = args[1:]
    elif len(args) == 0 or len(args) == 4:
        # No renderer is given. Create a new one and set its properties
        # to the given values or global defaults. The renderer becomes
        # the active renderer.
        _renderer = vtk.vtkRenderer()
        _renderer.SetUseDepthPeeling(1)
        _renderer.SetOcclusionRatio(0.1)
        _renderer.SetMaximumNumberOfPeels(10)
        _renderer.SetUseFXAA(1)

        # Add the new renderer to the list of all renderers of the main
        # render window. Otherwise the generated list is consumed when
        # the main window instance is created in show().
        if _renwin is not None:
            _renwin.AddRenderer(_renderer)
        else:
            _renderers.append(_renderer)

        # Default camera orientation. Overwritten later if the camera
        # argument is given.
        cam = _renderer.GetActiveCamera()
        cam.SetPosition(1.0, 1.0, 0.3)
        cam.SetViewUp(0.0, 0.0, 1.0)

        # Assign default color for new renderers if no color is given.
        if color is None:
            _renderer.SetBackground(colors.white)
    else:
        raise ValueError('wrong number of positional arguments')

    if len(args) == 4:
        _renderer.SetViewport(args[0], args[2],             # lower left
                              args[1], args[3])             # upper right

    if camera is not None:
        # Set the provided camera. Either directly or use the camera of
        # another renderer. The latter case will sync the viewports.
        if isinstance(camera, vtk.vtkRenderer):
            camera = camera.GetActiveCamera()

        _renderer.SetActiveCamera(camera)

    if color is not None:
        _renderer.SetBackground(color)

    if transparent is not None:
        _renderer.SetPreserveColorBuffer(transparent)

    if interactive is not None:
        _renderer.SetInteractive(interactive)

    # Viewports can be layered, i.e., occupy the same region in a render
    # window or overlap partially. In this case it can be useful to set
    # all viewports except for the bottom one as transparent (this is
    # VTK's default behavior when using the `layer` argument).
    if layer is not None:
        # The render window to which this renderer belongs needs to
        # support multiple layers for this to work, see show() and the
        # method SetNumberOfLayers() of a render window.
        _renderer.SetLayer(layer)

    # A None value will not touch values for the top background color
    # and gradient background setting. A False value disables gradient
    # background for this renderer.
    if color2 is False:
        _renderer.SetGradientBackground(False)
    elif color2 is not None:
        _renderer.SetBackground2(color2)
        _renderer.SetGradientBackground(True)

    return _renderer


def _window(width=1200, height=600, title=None, color=colors.white,
           color2=None, camera=None, interactive=True):
    """ Create render window (experimental).

    Create a secondary window. Secondary windows are limited to a single
    viewport whose properties are set during window creation.

    Parameters
    ----------
    width : int, optional
        Window width in pixels.
    height : int, optional
        Window height in pixels.
    title : str, optional
        Window title.
    color : array_like, shape (3, ), optional
        Window background color.
    color2 : array_like, shape (3, ), optional
        Top background color for gradient background.
    camera : vtkRenderer or vtkCamera, optional
        Useful to sync the view of different renderers.
    interactive : bool, optional
        Toggle event notification for the window's viewport.

    Returns
    -------
    vtkRenderWindow
        The newly created render window.

    Note
    ----
    Using secondary windows can cause segmentation faults depending on the
    used operating system and VTK version. Additionally secondary windows
    have to be closed using the keyboard shortcut 'x' instead of the close
    button.
    """
    # Create viewport that spans the entire render window. Windows created
    # by this function are limited to one renderer. Such renderers do not
    # go to the list _renderers of all renderers.
    ren = vtk.vtkRenderer()
    ren.SetUseDepthPeeling(1)
    ren.SetOcclusionRatio(0.1)
    ren.SetMaximumNumberOfPeels(10)
    ren.SetUseFXAA(1)

    # Apply user defined customization to the look and feel of the window.
    ren.SetBackground(color)
    ren.SetInteractive(interactive)

    if color2 is not None:
        ren.SetBackground2(color2)
        ren.SetGradientBackground(True)

    if camera is not None:
        # Set the provided camera. Either directly or use the camera
        # of another renderer. The latter case will sync the viewports.
        if isinstance(camera, vtk.vtkRenderer):
            camera = camera.GetActiveCamera()

        ren.SetActiveCamera(camera)
    else:
        # Change the camera position. It will still look at the origin
        # of the world coordinate system.
        cam = ren.GetActiveCamera()
        cam.SetPosition(1.0, 1.0, 0.3)
        cam.SetViewUp(0.0, 0.0, 1.0)

    # Create a window, set its size and title. Multisampling is turned
    # off because of transparent objects.
    renwin = vtk.vtkRenderWindow()
    renwin.SetSize(width, height)
    renwin.SetWindowName(str(title))
    renwin.SetMultiSamples(0)
    renwin.SetAlphaBitPlanes(1)
    renwin.AddRenderer(ren)

    # Seems to be too early to call this method. Results in segmentation
    # faults on some VTK implementations.
    # renwin.Render()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)
    iren.SetInteractorStyle(_MouseInteractorStyle())

    # Apparently this should be the last call when creating a new window.
    # iren.Initialize()

    # Add to the list of all interactors. This interactors will never
    # control the main render window.
    _interactors.append(iren)

    # Calling this method later does not place the axes_widget at the
    # correct spot.
    renwin.Render()

    # Set application window taskbar and dock icon.
    _app_icon(renwin)

    # Camera orientation widget was introduced in VTK 9.1, alterantively
    # we can show an axis widget.
    if (vtk.vtkVersion.GetVTKMajorVersion() > 8
            and vtk.vtkVersion.GetVTKMinorVersion() > 0):
        # A reference to the widget has to be maintained to prevent it
        # from being garbage collected immediately.
        iren.axes_widget = vtk.vtkCameraOrientationWidget()
        iren.axes_widget.SetParentRenderer(ren)
        iren.axes_widget.GetRepresentation().SetPadding(40, 40)
        iren.axes_widget.SquareResize()
        iren.axes_widget.SetEnabled(True)
    else:
        # Axes display in the lower left corner of each viewport. Only
        # shown in the active viewport, i.e., the one with mouse focus.
        iren.axes_widget = vtk.vtkOrientationMarkerWidget()
        iren.axes_widget.SetOrientationMarker(vtk.vtkAxesActor())
        iren.axes_widget.SetCurrentRenderer(ren)
        iren.axes_widget.SetInteractor(iren)
        iren.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        iren.axes_widget.SetSizeConstraintDimensionSizes(128, 256)
        # iren.axes_widget.SetShouldConstrainSize(True)
        iren.axes_widget.SetInteractive(False)
        iren.axes_widget.SetEnabled(True)

    # Initialize the render window interactor. Appears that only one
    # interactor has to be initialized/started?!
    # iren.Initialize()

    return renwin


def add(obj, renderer=None):
    """ Queue object for display.

    The `renderer` argument should be a value returned by :func:`canvas`
    or a window instance returned by :func:`window`. If not specified,
    the current viewport is used or a new one is created if this is the
    first object to be displayed.

    Parameters
    ----------
    obj : Prop or vtkActor
        Instance of a render object.
    renderer : vtkRenderer or vtkRenderWindow, optional
        The viewport or window to display the object.

    Returns
    -------
    vtkRenderer
        The renderer instance used for display.

    Note
    ----
    It is almost never necessary to use this function directly. It is
    used by all drawing and plotting commands automatically.
    """
    if renderer is not None:
        if isinstance(renderer, vtk.vtkRenderWindow):
            # It is assumed that this window was created by window().
            # The actor get placed on its unique renderer.
            renderer = renderer.GetRenderers().GetFirstRenderer()

        assert isinstance(renderer, vtk.vtkRenderer)
    else:
        # No renderer is given. Either create a new one or use the current
        # renderer if there is one.
        if _renderer is None:
            # Create initial renderer. Sets the global _renderer variable
            # and caches the renderer to be processed by show() later.
            canvas()

        # The global _renderer variable has been updated by canvas() and
        # no longer holds a None value.
        renderer = _renderer

    # Get the wrapped vtkProp instance. This is the most general object
    # type that can be added to the scene.
    prop = obj.prop if isinstance(obj, Prop) else obj
    renderer.AddViewProp(prop)

    return renderer


def aabb(points, opacity=0.15, edges=True, labels='dim', color=colors.snow):
    r""" Axis aligned bounding box.

    Display axis aligned bounding box of `points` with annotation.

    Parameters
    ----------
    points : Prop or array_like, shape (3, k)
        An array of :math:`k` points in 3-space.
    opacity : float, optional
        Opacity of bounding box.
    edges : bool, optional
        Toggle edges of the bounding box.
    labels : str, optional
        Either :obj:`None`, 'dim', 'minmax', or 'both'.
    color : array_like, shape (3, ), optional
        Bounding box color.

    Returns
    -------
    PolyMesh
        Bounding box shape.

    Note
    ----
    Visual properties of the bounding box edges can be further customized
    via the returned :class:`PolyMesh` instance.
    """
    if isinstance(points, Prop):
        a, b = points.bounds
    else:
        a = np.min(points, axis=0)
        b = np.max(points, axis=0)

    if labels == 'both' or labels == 'minmax':
        label(a, f'({a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f})', size=18)
        label(b, f'({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f})', size=18)

    if labels == 'both' or labels == 'dim':
        scale = 0.075 * np.min(b-a)
        dx, dy, dz = b-a

        _put([b[0], b[1], a[2]], dx, f'{dx:.2f}', scale, axis='-x')
        _put([b[0], a[1], a[2]], dy, f'{dy:.2f}', scale, axis='y')
        _put([a[0], b[1], a[2]], dz, f'{dz:.2f}', scale, axis='z')

    cube = vtk.vtkCubeSource()
    cube.SetBounds(a[0], b[0],
                   a[1], b[1],
                   a[2], b[2])
    cube.Update()

    box = PolyMesh(cube.GetOutput())
    box.color = color
    box.opacity = opacity

    if edges:
        box.edges(style='lines', width=2, color=colors.ivory_black)

    add(box)
    return box


def delete(*actors, renderer=None):
    """ Remove actor(s).

    Search all defined viewports and remove the given actor(s).

    Parameters
    ----------
    *actors
        Sequence of actors to be removed.
    renderer : vtkRenderer, optional
        Remove actors only from this renderer.

    Note
    ----
    This will prevent an object from being displayed. As long as there are
    other references to it, it will not be removed from memory.
    """
    if renderer is None:
        for iren in _interactors:
            renwin = iren.GetRenderWindow()
            renderers = renwin.GetRenderers()

            for renderer in renderers:
                for actor in actors:
                    if isinstance(actor, Prop):
                        actor = actor.prop

                    renderer.RemoveViewProp(actor)
    else:
        for actor in actors:
            if isinstance(actor, Prop):
                actor = actor.prop

            renderer.RemoveViewProp(actor)


def update(*args, **kwargs):
    """ Update viewports.

    Some changes to actors require an explicit re-render requests. Use this
    function if your changes are not diplayed properly. Changes applied
    to external data structures that are sourced by VTK objects have to be
    communicated by invoking the corresponding :meth:`modified` method.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Dictionary of keyword arguments.

    Note
    ----
    This functions defines arguments (which are ignored) to make it directly
    usable as a callback function.
    """
    # Calling update() before a render window is created results in an
    # exception. Since a forced update is pointless in this situation, we
    # can safely ignore this problem.
    for iren in _interactors:
        iren.Render()


def screenshot(filename, scale=1.0, window=None):
    """ Save the current framebuffer contents.

    Parameters
    ----------
    filename : str
        Name of PNG target file.
    scale : float, optional
        Scale factor.
    window : vtkRenderWindow, optional
        Window to grab.

    Note
    ----
    Some display properties (for example the width of mesh edges) are given
    in screen units (i.e., pixels). This may give some unexpected results when
    using the `scale` argument to change the resolution of the saved image.
    """
    window = _renwin if window is None else window
    width, height = window.GetSize()

    filter = vtk.vtkResizingWindowToImageFilter()
    filter.SetInput(window)
    filter.SetInputBufferTypeToRGBA()
    filter.SetSize(int(scale*width), int(scale*height))
    filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(filter.GetOutputPort())
    writer.Write()


def _test():
    n = 100

    points = np.random.rand(n, 3) - 0.5
    vectors = np.random.rand(n, 3) - 0.5
    angles = np.pi * (0.1 + 0.3 * np.random.rand(n))
    radius = np.random.rand(n) * 0.1
    size = 0.05 * np.random.rand(n)
    color = np.random.rand(n)
    color3 = np.random.rand(n, 3)
    # color3 = np.zeros((n, 3))
    # color3[:, 0] = 1.0

    # glyphs = scatter(points, size=0.5, color=colors.cornflower)
    # glyphs.colorize(color)
    # glyphs.opacity = 0.5

    glyphs = splat(points, vectors, size=0.1)
    glyphs.colorize(color)
    glyphs.lookuptable((np.min(color), np.max(color)))
    # glyphs.scale(color)

    colorbar(glyphs)
    aabb(glyphs)

    # points = np.array([[0., 0., 0.],
    #                    [1., 0., 0.],
    #                    [2., 0., 0.]])

    # vectors = np.array([[0., 0., 1.],
    #                     [0., 0., 1.],
    #                     [0., 0., 1.]])

    # angles = np.array([0.4 * np.pi, 0.25 * np.pi, 0.1 * np.pi])
    # height = np.array([0.1, 0.1, 0.1])

    # glyph = cones(points, vectors, angles, radius=radius, double=True)
    # glyph.colorize(angles)

    # colorbar(glyph)
    # aabb(glyph)

    show()


def colorbar(object, x=0.8, y=0.1):
    """ Display colorbar.

    Display visual representation of the lookup table associated with
    `object`.

    Parameters
    ----------
    object : Prop or vtkActor
        A render object.
    x : float, optional
        Horizontal position in normalized window coordinates.
    y : float, optional
        Vertical position in normalized window coordinates.

    Returns
    -------
    LookupTable
        Visual representation of lookup table.
    """
    colorbar = LookupTable(object)
    colorbar.position = x, y

    add(colorbar)
    return colorbar


def scatter(points, style='spheres', size=1.0, color=colors.dim_grey):
    """ Scatter plot.

    Visual representation of `points`. Display spheres of given radius.

    Parameters
    ----------
    points : array_like, shape (n, 3), n > 1
        Point coordinates.
    style : str, optional
        Either 'spheres' or 'cubes'.
    size : float, optional
        Sphere radius.
    color : array_like, shape (3, ), optional
        Color intensity triplet.

    Returns
    -------
    Spheres
        Sphere congruence prop.

    Note
    ----
    If `points` is of type :class:`~numpy.ndarray` its data buffer is
    shared with VTK's data objects (use a copy of `points` to decouple
    storage).
    """
    pc = Spheres(points)

    pc.style = style
    pc.size = size
    pc.color = color

    add(pc)
    return pc


def splat(points, vectors, size=1.0, color=colors.snow):
    """ Point splatting.

    Display disk at locations orthogonal to given directions.

    Parameters
    ----------
    points : array_like
        Point locations.
    vectors : array_like
        Direction vectors.
    size : float, optional
        Disk radius.
    color : array_like, shape (3, ), optional
        Color specification.
    """
    splats = Disks(points, vectors)

    splats.size = size
    splats.color = color

    add(splats)
    return splats


def quiver(points, vectors, size=1.0, color=colors.green_pale, **kwargs):
        #    shaft_radius=0.025, tip_radius=0.05, tip_length=0.5,
        #    resolution=6):
    """ Quiver plot.

    Display arrows at given locations pointing in given directions. For
    each point exactly one direction vector has to be given.

    Parameters
    ----------
    points : array_like, shape (k, 3)
        Point coordinates.
    vectors : array_like, shape (k, 3)
        Vector coordinates.
    size : float, optional
        Length of the displayed arrows.
    color : array_like, optional
        Color specification.

    Returns
    -------
    Arrows
        Vector field prop.


    The glyphs used to model arrows can be customized via the following
    keyword arguments.

    Keyword arguments
    -----------------
    shaft_radius : float, optional
    tip_radius : float, optional
    tip_length : float, optional
    resolution : int, optional

    Note
    ----
    If `array_like` parameters are of type :class:`~numpy.ndarray` their
    data buffer is shared with VTK's data objects (use copies to decouple
    storage).
    """
    vf = Arrows(points, vectors, **kwargs)

    vf.size = size
    vf.color = color

    add(vf)
    return vf


def _generic_lut(range=(0.0, 1.0), gradient='default', logscale=False,
                 size=128, below=None, above=None, nan=None):
    """ Generate lookup table.

    Lookup table with `size` values in the given `range`. Gradient
    values drawn from

        {'default', 'hot', 'jet', 'grey'}

    generate a continuous color gradient whereas values from

        {'spectral', 'diverging', 'blue', 'orange', 'purple'}

    only define a fixed number of discrete colors. The `size` parameter
    has no effect in the latter case. Special colors can be assigned to
    values outside the specified range.

    Parameters
    ----------
    range : (float, float), optional
        Range of table values.
    gradient : str, optional
        Name of color gradient.
    logscale : bool, optional
        Toggle logarithmic scaling.
    size : int, optional
        Number of table values.
    below : array_like, shape (4, )
        RGBA color for values below table range.
    above : array_like, shape (4, )
        RGBA color for values above table range.
    nan : array_like, shape (4, )
        RGBA color for NaN values.

    Returns
    -------
    vtkLookupTable
        The generated lookup table.

    Note
    ----
    Passing 'default' as `gradient` will use VTK's default color gradient.
    """
    if gradient in {'default', 'hot', 'jet', 'grey', 'gray'}:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(size)

        if gradient == 'hot':
            lut.SetHueRange(0, 1/6)
            lut.SetSaturationRange(1, 0.5)
            lut.SetValueRange(1, 1)
        elif gradient == 'jet':
            lut.SetHueRange(2/3, 0)
            lut.SetSaturationRange(1, 1)
            lut.SetValueRange(1, 1)
        elif gradient == 'grey' or gradient == 'gray':
            lut.SetHueRange(0, 0)
            lut.SetSaturationRange(0, 0)
            lut.SetValueRange(0, 1)

        lut.Build()
    elif gradient in {'spectral', 'diverging', 'blue', 'orange', 'purple'}:
        series = vtk.vtkColorSeries()

        map = {'spectral': series.BREWER_DIVERGING_SPECTRAL_11,
               'diverging': series.BREWER_DIVERGING_BROWN_BLUE_GREEN_10,
               'blue': series.BREWER_SEQUENTIAL_BLUE_GREEN_9,
               'orange': series.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9,
               'purple': series.BREWER_SEQUENTIAL_BLUE_PURPLE_9}

        series.SetColorScheme(map[gradient])
        lut = series.CreateLookupTable(series.ORDINAL)
    else:
        raise ValueError(f"unknown color scheme '{gradient}'")

    lut.SetTableRange(range[0], range[1])

    if logscale:
        lut.SetScaleToLog10()

    if nan is not None:
        lut.SetNanColor(nan)

    if below is not None:
        lut.SetBelowRangeColor(below)
        lut.SetUseBelowRangeColor(True)

    if above is not None:
        lut.SetAboveRangeColor(above)
        lut.SetUseAboveRangeColor(True)

    return lut


def _tweak_lut(lut, range=None, gradient=None, logscale=None, size=None,
               **kwargs):
    """ Modify lookup table properties.

    An objects lookup tables determines how entries of the scalar array
    are translated to color values. Lookup tables have no effect when
    directly mapping RGB color values.

    Parameters
    ----------
    range : (float, float)
        Accpeted range of scalar values.
    gradient : str
        Color scheme identifier, see below.
    logscale : bool
        Switch between linear and logarithmic scale.
    size : int
        Size of lookup table.

    Keyword arguments
    -----------------
    below : array_like, shape (4, )
        Color for scalars below the specified range.
    above: array_like, shape (4, )
        Color for scalars above the specified range.
    nan : array_like, shape (4, )
        Special color for NaN scalar values.


    Smooth color gradients are defined by the color scheme identifiers
    'hot', 'cool', 'jet', and 'grey'. The color schemes 'spectral',
    'diverging', 'blue', 'orange', and 'purple' define a discrete color
    series.

    If provied, out of range values are marked with the `below`,
    `above`, and `nan` colors. Not that those colors also have an alpha
    intensity value to control opacity.

    Note
    ----
    Arguments not provided have no affect on the corresponding lookup
    table property.
    """
    if gradient in {'hot', 'cool', 'jet', 'grey', 'gray'}:
        if gradient == 'hot':
            lut.SetHueRange(0, 1/6)
            lut.SetSaturationRange(1, 0.5)
            lut.SetValueRange(1, 1)
        elif gradient == 'cool':
            lut.SetHueRange(0.49, 0.6)
            lut.SetSaturationRange(1, 0.1)
            lut.SetValueRange(0.55, 1)
        elif gradient == 'jet':
            lut.SetHueRange(2/3, 0)
            lut.SetSaturationRange(1, 1)
            lut.SetValueRange(1, 1)
        elif gradient == 'grey' or gradient == 'gray':
            lut.SetHueRange(0, 0)
            lut.SetSaturationRange(0, 0)
            lut.SetValueRange(0, 1)

        lut.ForceBuild()
    elif gradient in {'spectral', 'diverging', 'blue', 'orange', 'purple'}:
        # Color series define a fixed number of colors. A given size
        # parameter is ignored in this case.
        series = vtk.vtkColorSeries()

        map = {'spectral': series.BREWER_DIVERGING_SPECTRAL_11,
               'diverging': series.BREWER_DIVERGING_BROWN_BLUE_GREEN_10,
               'blue': series.BREWER_SEQUENTIAL_BLUE_GREEN_9,
               'orange': series.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9,
               'purple': series.BREWER_SEQUENTIAL_BLUE_PURPLE_9}

        series.SetColorScheme(map[gradient])
        series.BuildLookupTable(lut, series.ORDINAL)
    elif gradient == 'default':
        # This should reset the color gradient to some default value.
        # Currently does nothing.
        pass
    elif gradient is None:
        # Nothing to do, no gradient argument defined.
        pass
    else:
        raise ValueError(f"unknown color scheme '{gradient}'")

    if range is not None:
        lut.SetTableRange(range[0], range[1])

    if logscale is not None:
        if logscale:
            lut.SetScaleToLog10()
        else:
            lut.SetScaleToLinear()

    if size is not None:
        # Setting the size of a lookup table has no effect when using
        # a discrete color scheme.
        lut.SetNumberOfTableValues(size)

    # Not passing a keyword argument will not alter the corresponding
    # setting. Passing a None value disables the corresponding feature.
    if 'below' in kwargs:
        if (below := kwargs['below']) is not None:
            color = [0., 0., 0., 1.]
            color[:len(below)] = below

            lut.SetBelowRangeColor(color)
            lut.SetUseBelowRangeColor(True)
        else:
            lut.SetUseBelowRangeColor(False)

    if 'above' in kwargs:
        if (above := kwargs['above']) is not None:
            color = [0., 0., 0., 1.]
            color[:len(above)] = above

            lut.SetBelowRangeColor(color)
            lut.SetUseAboveRangeColor(True)
        else:
            lut.SetUseAboveRangeColor(False)

    if (nan := kwargs.get('nan')) is not None:
        color = [0., 0., 0., .5]
        color[:len(nan)] = nan

        lut.SetNanColor(color)
    # else:
    #     lut.SetNanColor(None)


def _app_icon(window, file='m3sh.png'):
    """
    """
    reader_factory = vtk.vtkImageReader2Factory()
    reader_factory.SetGlobalWarningDisplay(False)
    reader = reader_factory.CreateImageReader2(file)

    if reader is not None:
        reader.SetFileName(file)
        reader.Update()

        window.SetIcon(reader.GetOutput())


def mesh(mesh, color=colors.snow):
    """ Mesh visualization.

    Visualization of a polygonal mesh represented as a :class:`Mesh`
    instance or a 2-tuple holding an `array_like` vertex coordinate
    representation and a list of face definitions. The visual
    appearance of a mesh can be modified by using the methods and
    attributes of the returned object.

    Parameters
    ----------
    mesh : Mesh or tuple
        Polygonal mesh representation.
    color : array_like, shape (3, ), optional
        RGB color triple.

    Returns
    -------
    PolyMesh
        Polygonal mesh instance.
    """
    renmesh = PolyMesh(mesh)
    renmesh.color = color

    add(renmesh)
    return renmesh


def graph(nodes, graph, color=colors.black):
    """ Graph visualization.

    Experimental support for graph plotting. The `graph` parameter
    is described in more detail in SciPy's :mod:`~scipy.sparse.csgraph`
    module

    Parameters
    ----------
    nodes : array_like
        Node coordinate array.
    graph : csr_matrix
        Combinatorial graph definition.
    color : array_like, shape (3, ), optional
        Edge color.

    Returns
    -------
    PolyGraph
        Polygonal graph instance.

    Note
    ----
    Conversion from a dense matrix repesentation can be achieved
    via the :func:`~scipy.sparse.csgraph.csgraph_from_dense` utility
    function.
    """
    rengraph = PolyGraph(nodes, graph)
    rengraph.color = color

    add(rengraph)
    return rengraph


def _contour(M, S, levels=10, width=2.0, style='-', color=(0.0, 0.0, 0.0)):
    """ Contour plot.

    Parameters
    ----------
    M
        Either indexed or halfedge based mesh representation.
    S : array_like, shape (n, )
        A scalar value for each vertex of the mesh.
    levels : int, optional
        Number of level sets to generate.
    width : float, optional
        Line width in screen units.
    style : str, optional
        Line style used to display level sets.
    color : array_like, shape (3, ), optional
        Global RGB intensity triplet used as line color.

    Returns
    -------
    vtkActor
        Corresponding actor object.
    """
    # Point and face array setup. Vertex coordinates and face definitions are
    # always given.
    point_array = vtk.vtkPoints()
    face_array = vtk.vtkCellArray()

    # Detect the input type. A halfedge mesh M or and indexed mesh M = (V,F)
    # is expected as input.
    try:
        M.nverts
    except AttributeError:
        try:
            V = M[0]
            F = M[1]
        except (TypeError, IndexError):
            msg = ('mesh(): expecting indexed mesh (V, F) or halfedge ' +
                   'mesh M as input!')
            raise ValueError(msg)
        else:
            # The indexed mesh case. Fill vertex and face arrays. Texture
            # information on the face level is ignored.
            for v in V:
                try:
                    point_array.InsertNextPoint(v[0], v[1], v[2])
                except IndexError:
                    point_array.InsertNextPoint(v[0], v[1], 0.0)

            for f in F:
                face = vtk.vtkIdList()
                for vdef in f:
                    try:
                        face.InsertNextId(vdef[0])
                    except (TypeError, IndexError):
                        face.InsertNextId(vdef)
                face_array.InsertNextCell(face)
    else:
        # The halfedge mesh case. The variables nverts and nfaces have
        # already been set.
        for v in M.vertices():
            p = v.point
            try:
                point_array.InsertNextPoint(p[0], p[1], p[2])
            except IndexError:
                point_array.InsertNextPoint(p[0], p[1], 0.0)

        for f in M.faces():
            face = vtk.vtkIdList()
            for v in f.vertices():
                face.InsertNextId(v.index)
            face_array.InsertNextCell(face)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(point_array)
    polyData.SetPolys(face_array)

    float_array = vtk.vtkFloatArray()
    float_array.SetNumberOfComponents(1)

    for val in S:
        float_array.InsertNextTuple1(val)

    polyData.GetPointData().SetScalars(float_array)

    contour = vtk.vtkContourFilter()
    contour.SetInputData(polyData)
    contour.GenerateValues(levels, np.min(S), np.max(S))

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetColor(color[0], color[1], color[2])

    if '=' in style:
        actor.GetProperty().SetRenderLinesAsTubes(True)

    add(actor)
    return actor


def _silhouette(M):
    """
    """
    # Point and face array setup. Vertex coordinates and face definitions are
    # always given.
    point_array = vtk.vtkPoints()
    face_array = vtk.vtkCellArray()

    # Detect the input type. A halfedge mesh M or and indexed mesh M = (V,F)
    # is expected as input.
    try:
        nverts = M.nvertices
        nfaces = M.nfaces
    except AttributeError:
        try:
            V = M[0]
            F = M[1]
        except (TypeError, IndexError):
            msg = ('mesh(): expecting indexed mesh (V, F) or halfedge ' +
                   'mesh M as input!')
            raise ValueError(msg)
        else:
            for v in V:
                try:
                    point_array.InsertNextPoint(v[0], v[1], v[2])
                except IndexError:
                    point_array.InsertNextPoint(v[0], v[1], 0.0)

            for f in F:
                face = vtk.vtkIdList()
                for vdef in f:
                    try:
                        face.InsertNextId(vdef[0])
                    except (TypeError, IndexError):
                        face.InsertNextId(vdef)
                face_array.InsertNextCell(face)

            nverts = len(V)
            nfaces = len(F)
    else:
        for v in M.vertices():
            p = v.point
            try:
                point_array.InsertNextPoint(p[0], p[1], p[2])
            except IndexError:
                point_array.InsertNextPoint(p[0], p[1], 0.0)

        for f in M.faces():
            if not f.isdeleted():
                face = vtk.vtkIdList()
                for v in f.vertices():
                    face.InsertNextId(v.index)
                face_array.InsertNextCell(face)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(point_array)
    polyData.SetPolys(face_array)

    silhouette = vtk.vtkPolyDataSilhouette()
    silhouette.SetInputData(polyData)
    silhouette.SetCamera(_renderer.GetActiveCamera())
    # silhouette->SetEnableFeatureAngle(0);

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(silhouette.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    actor.GetProperty().SetLineWidth(1)

    add(actor)
    return actor


def _cones(points, vectors, angle=None, radius=None, height=None,
          color=colors.snow, double=False, cap=False, resolution=24):
    """ Cone rendering.

    Render cones with apices at the specified locations using the given
    vectors as cone axis. Cone shapes are defined by two sequences
    specifiying either cone angles, base radius, or cone height.

    Parameters
    ----------
    points : array_like, shape (n, 3)
        Array of cone apices.
    vectors : array_like, shape (n, 3)
        Array of cone axes.
    angle : array_like of shape (n, ) or float
    radius : array_like of shape (n, ) or float
    height : array_like of shape (n, ) or float
    color : array_like, shape (3, ), optional
        Object color.
    double : bool, optional
        :obj:`True` to generate double cones.
    cap : bool, optional
        :obj:`True` to close cones along their base.
    resolution : int, optional
        Number of vertices of base polygon.

    Returns
    -------
    ConeField
        The generated assembly of cones.

    Note
    ----
    In contrast to cone geometry, color cannot be specified on a per cone
    level, there is only one color for all cones in the assembly.
    """
    if len(points) != len(vectors):
        raise ValueError("array shapes don't match")

    if angle is not None:
        if np.ndim(angle) == 0:
            angle = len(points) * [angle]
        else:
            if len(angle) != len(points):
                raise ValueError("array shapes don't match")

        # If there is an angle greater than 90 degrees the corresponding
        # axis should be inverted and the angle replaced by 180 - angle.
        for i, alpha in enumerate(angle):
            if alpha > 0.5 * np.pi:
                raise ValueError(f'angle out of range, got {alpha}')

                # This fix messes with the input data and produces side
                # effects. The data should be prepared properly before
                # visualization.
                angle[i] = np.pi - alpha
                vectors[i] *= -1.0
            elif alpha < 0.0:
                raise ValueError('angle has to be non-negative')

    if radius is not None:
        if np.ndim(radius) == 0:
            radius = len(points) * [radius]
        else:
            if len(radius) != len(points):
                raise ValueError("array shapes don't match")

    if height is not None:
        if np.ndim(height) == 0:
            height = len(points) * [height]
        else:
            if len(height) != len(points):
                raise ValueError("array shapes don't match")

    conefield = _ConeField(points, vectors, angle, radius, height,
                          double, cap, resolution)
    conefield.color = color

    add(conefield.prop)
    return conefield


def cones(points, vectors, angle=None, radius=None, height=None,
          color=colors.snow, double=False, cap=False, resolution=24):
    """ Cone rendering.
    """
    if len(points) != len(vectors):
        raise ValueError("array shapes don't match")

    if angle is not None:
        if np.ndim(angle) == 0:
            angle = len(points) * [angle]
        else:
            if len(angle) != len(points):
                raise ValueError("array shapes don't match")

        # If there is an angle greater than 90 degrees the corresponding
        # axis should be inverted and the angle replaced by 180 - angle.
        for i, alpha in enumerate(angle):
            if alpha > 0.5 * np.pi:
                raise ValueError(f'angle out of range, got {alpha}')

                # This fix messes with the input data and produces side
                # effects. The data should be prepared properly before
                # visualization.
                angle[i] = np.pi - alpha
                vectors[i] *= -1.0
            elif alpha < 0.0:
                raise ValueError('angle has to be non-negative')

    if radius is not None:
        if np.ndim(radius) == 0:
            radius = len(points) * [radius]
        else:
            if len(radius) != len(points):
                raise ValueError("array shapes don't match")

    if height is not None:
        if np.ndim(height) == 0:
            height = len(points) * [height]
        else:
            if len(height) != len(points):
                raise ValueError("array shapes don't match")

    cones = Cones(points, vectors, angle, radius, height, double,
                  cap, resolution)

    cones.color = color

    add(cones)
    return cones


def _plot(points):
    pass


def plot(P, width=2.0, size=6.0, style='-', color=(0.25, 0.25, 0.25)):
    """ Polyline plotting.

    Parameters
    ----------
    P : array_like, shape (n, 3)
        Coordinate array of vertices.
    width : float
        Width of line segments in screen units.
    size: float
        Marker size in screen units.
    style : string
        A combination of '.', 'o', '-', and '='.
    color : array_like, shape (3, )
        RGB color triplet for all line segments and markers.

    Returns
    -------
    vtkActor
        The corresponding actor object.
    """
    idx = len(P) * [-1]
    j = 0

    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    for i, p in enumerate(P):
        if not np.any(np.isnan(p)):
            points.InsertNextPoint(*p)

            vert = vtk.vtkIdList()
            vert.InsertNextId(j)
            verts.InsertNextCell(vert)

            idx[i] = j
            j += 1

    lines = vtk.vtkCellArray()

    for i in range(len(P)-1):
        if idx[i] > -1 and idx[i+1] > -1:
            line = vtk.vtkIdList()
            line.InsertNextId(idx[i])
            line.InsertNextId(idx[i+1])
            lines.InsertNextCell(line)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)

    if ('o' in style) or ('.' in style): polyData.SetVerts(verts)
    if ('-' in style) or ('=' in style): polyData.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetPointSize(size)
    actor.GetProperty().SetColor(color[0], color[1], color[2])

    if 'o' in style: actor.GetProperty().SetRenderPointsAsSpheres(True)
    if '=' in style: actor.GetProperty().SetRenderLinesAsTubes(True)

    add(actor)
    return actor


def box(a, b, opacity=1.0, edges=True, color=colors.snow):
    r""" Box shape.

    Display the cuboid
    :math:`Q = [a_x, b_x] \times [a_y, b_y] \times [a_z, b_z]`.

    Parameters
    ----------
    a : array_like, shape (3, )
        Min corner of the box.
    b : array_like, shape (3, )
        Max corner of the box.
    opacity : float, optional
        Opacity of box.
    edges : bool, optional
        Toggle edges of the box.
    color : array_like, shape (3, ), optional
        RGB color triplet.

    Returns
    -------
    PolyMesh
        The corresponding polygonal shape representation.

    Note
    ----
    Visual properties of the box edges can be further customized via
    the returned :class:`PolyMesh` instance.
    """
    cube = vtk.vtkCubeSource()
    cube.SetBounds(a[0], b[0],
                   a[1], b[1],
                   a[2], b[2])
    cube.Update()

    box = PolyMesh(cube.GetOutput())
    box.color = color
    box.opacity = opacity

    if edges:
        box.edges('lines')

    add(box)
    return box


def _circle(center, normal, radius):
    """
    """
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(center))

    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.GetPointData().SetNormals(numpy_to_vtk(normal))
    data.GetPointData().SetScalars(numpy_to_vtk(radius))

    circle = vtk.vtkArcSource()
    circle.SetCenter(0.0, 0.0, 0.0)
    circle.SetNormal(1.0, 0.0, 0.0)
    circle.SetPolarVector(0.0, 1.0, 0.0)
    circle.SetAngle(360.0)
    circle.SetUseNormalAndAngle(True)
    circle.SetResolution(60)
    # circle.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(data)
    glyph.SetSourceConnection(circle.GetOutputPort())

    glyph.OrientOn()
    glyph.SetVectorModeToUseNormal()

    glyph.ScalingOn()
    glyph.SetScaleModeToScaleByScalar()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)

    add(actor)
    return actor, data


def cylinder(a, b, radius, opacity=1.0, resolution=64, cap=False,
             color=colors.grey):
    """ Cylinder shape.

    Display cylinder with axis starting at point `a` and ending at
    point `b`

    Parameters
    ----------
    a : array_like, shape (3, )
        First point on cylinder axis.
    b : array_like, shape (3, )
        Last point on cylinder axis.
    radius : float
        Cylinder radius.
    opacity : float, optional
        Cylinder opacity.
    resolution : int, optional
        Number of sides.
    cap : bool, optional
        If :obj:`True` the cylinder is closed.
    color : array_like, shape (3, ), optional
        RGB color triplet.

    Returns
    -------
    PolyData
        The corresponding polygonal shape representation.
    """
    line = vtk.vtkLineSource()
    line.SetPoint1(a[0], a[1], a[2])
    line.SetPoint2(b[0], b[1], b[2])

    filter = vtk.vtkTubeFilter()
    filter.SetInputConnection(line.GetOutputPort())
    filter.SetRadius(radius)
    filter.SetNumberOfSides(resolution)
    filter.SetCapping(cap)
    # filter.SetSidesShareVertices(False)
    filter.Update()

    tube = PolyData(filter.GetOutput())
    tube.color = color
    tube.opacity = opacity

    add(tube)
    return tube


def sphere(center, radius, opacity=1.0, resolution=24, color=colors.snow):
    """ Sphere shape.

    Covenience function that displays a sphere with given center and
    radius.

    Parameters
    ----------
    center : array_like, shape (3, )
        Sphere center.
    radius : float
        Sphere radius.
    opacity : float, optional
        Opacity value.
    resolution : int, optional
        Angle discretization.
    color : array_like, shape (3, ), optional
        RGB color triplet.

    Returns
    -------
    PolyData
        The corresponding polygonal shape representation.
    """
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.Update()

    ball = PolyData(sphere.GetOutput())
    ball.color = color
    ball.opacity = opacity

    add(ball)
    return ball


def _spheres(C, r):
    """
    Visualize points at given (x,y,z) locations as spheres of a given radius
    and color. Unless you need differently sized point markers you should use
    scatter2() instead.

    Parameters
    ----------
    X, Y, Z : array_like
        Coordinate arrays need to be of the same shape.
    size : array_like
        A single scalar value used as radius for all spheres. To specify individual
        radii size needs to be of the same shape as the coordinate arrays.
    color : array_like
        Color intensity triplet(s). A single triplet specifies a global color
        for all points. To specify colors on a per point basis the last axis of
        col needs to hold color intensity triplets and each intensity channel
        col[..., i], i = 0,1,2, needs to be of the same shape as X, Y, and Z.
    scalar : array_like
        Scalar data associated with points, one scalar per point, i.e., the shape
        of the scalar data needs to match the match of X, Y, and Z. If specified
        color mapping based on scalar values takes precedence of direct coloring
        using the color array.
    cmap : string
        Color map identifier. Only used when scalar data is given. Invalid values
        prevents scalar based color mapping.

    Returns
    -------
    actor :
        A vtk actor.

    """
    # In case the input is given as lists of lists etc. Won't change anything
    # if the input are already ndarrays (no copying or other overhead incurred).
    # C = np.asarray(C)
    # r = np.asarray(r)

    # assert len(C) == len(r)

    # Prepare the colors and the corresponding mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetScalarModeToUsePointFieldData()

    # Build the vtk representation of the point cloud
    points = vtk.vtkPoints()
    for x in C:
        points.InsertNextPoint(x[0], x[1], x[2])

    radius = vtk.vtkFloatArray()
    radius.SetName('glyph_size')
    radius.SetNumberOfComponents(1)

    for x in r:
        radius.InsertNextTuple1(x)

    # Create a polydata to store everything in
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().AddArray(radius)
    polyData.GetPointData().SetActiveScalars('glyph_size')

    # The source shape used for glyphs. If rendering is too slow when there
    # is a large number of glyphs, the resolution of each sphere can be
    # reduced.
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(20)
    sphere.SetPhiResolution(15)
    sphere.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(polyData)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOn()
    glyph.SetScaleModeToScaleByScalar()

    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)

    add(actor)
    return actor


# def tree(boxes):
#     tree = BoxArray(boxes)
#     tree.opacity = 0.5

#     for actor in tree.prop.GetParts():
#         actor.GetProperty().SetEdgeVisibility(True)

#     add(tree)
#     return tree


def display(message, x=0.05, y=0.95, size=12, color=colors.white,
            bold=False, italic=False, shadow=False, frame=False):
    """ Display static text in the render window.

    Location of the displayed text is given in normalized window
    coordinates, i.e., relative to the width and height of the render
    window.

    Parameters
    ----------
    message : str
        The message to be displayed.
    x : float, optional
        Coordinate in the interval [0, 1].
    y : float, optional
        Coordinate in the interval [0, 1].
    size : int, optional
        Font size.

    Returns
    -------
    Prop
        The wrapped text actor.
    """
    text = vtk.vtkTextProperty()
    text.SetFontSize(size)
    text.SetFontFamilyToCourier()
    text.SetColor(color)
    text.SetBold(bold)
    text.SetBackgroundColor(colors.black)
    text.SetBackgroundOpacity(0.0)
    text.SetFrame(frame)
    text.SetItalic(italic)
    text.SetShadow(shadow)
    text.SetVerticalJustificationToTop()

    mapper = vtk.vtkTextMapper()
    mapper.SetInput(message)
    mapper.SetTextProperty(text)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.GetPositionCoordinate().SetValue(x, y)

    add(actor)
    return Prop(actor)


def _put(pos, dim, text, scale=1.0, opacity=0.75, axis='x', follow=False,
         color=colors.black):
    """ Text annotation.

    Puts the given text at a certain location. The text orientation
    remains static or it can follow the camera.

    Parameters
    ----------
    pos : array_like, shape (3, )
        Text location.
    dim : float
        Text dimension.
    text : str
        Text to be displayed.
    scale : float, optional
        Scale factor.
    opacity : float, optional
        Text opacity.
    axis : str, optional
        Text orientation, no effect when `follow` evaluates to :obj:`True`.
    follow : bool, optional
        :obj:`True` orients the text such that is always faces the camera.
    color : array_like, shape (3, ), optional
        RGB triplet of color intensities.

    Returns
    -------
    Actor
        The created :class:`Actor` instance.
    """
    vector_text = vtk.vtkVectorText()
    vector_text.SetText(text)
    vector_text.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vector_text.GetOutput())

    # An actor has its own GetBounds() method that takes the actor's
    # transformation into account.
    bounds = mapper.GetBounds()

    # In its untransformed state the text is aligned along the positive
    # x-axis in the z=0 plane starting at the origin.
    dx = scale * (bounds[1] - bounds[0])
    dy = scale * (bounds[3] - bounds[2])
    dz = scale * (bounds[5] - bounds[4])

    if follow:
        actor = vtk.vtkFollower()
        actor.SetMapper(mapper)
        actor.SetCamera(_renderer.GetActiveCamera())

        if axis == '-x':
            actor.SetPosition(pos[0] - 0.5*dim, pos[1], pos[2])
        elif axis == 'y':
            actor.SetPosition(pos[0], pos[1] + 0.5*dim, pos[2])
        elif axis == 'z':
            actor.SetPosition(pos[0], pos[1], pos[2] + 0.5*dim)
    else:
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if axis == '-x':
            actor.SetOrientation(90.0, 0.0, 180.0)
            actor.SetPosition(pos[0] - 0.5*(dim-dx), pos[1], pos[2])
        elif axis == 'y':
            actor.SetOrientation(90.0, 0.0, 90.0)
            actor.SetPosition(pos[0], pos[1] + 0.5*(dim-dx), pos[2])
        elif axis == 'z':
            actor.SetOrientation(0.0, -90.0, 270.0)
            actor.SetPosition(pos[0], pos[1], pos[2] + 0.5*(dim-dx))

    actor.SetScale(scale, scale, scale)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    # The vtkFollower has to be modelled as Actor. Otherwise we could
    # also use PolyData as final render object.
    add(actor)
    return Prop(actor)


def _caption(pos, text, size=2.0, opacity=0.0, color=(1.0, 1.0, 1.0)):
    """
    """
    textActor = vtk.vtkCaptionActor2D()
    textActor.SetAttachmentPoint(pos)
    textActor.SetCaption(text)
    textActor.SetBorder(False)
    textActor.SetThreeDimensionalLeader(True)
    textActor.SetLeader(False)
    textActor.SetPosition(-5.0, -5.0)
    textActor.SetWidth(0.01*size)
    textActor.SetHeight(0.01*size)
    textActor.SetPadding(0)
    # textActor.GetProperty().SetLineWidth(4)

    textProp = textActor.GetCaptionTextProperty()
    textProp.SetBackgroundColor(colors.black)
    textProp.SetBackgroundOpacity(opacity)
    textProp.SetColor(color)

    add(textActor)
    return textActor


def label(pos, text, size=12, opacity=1.0, color=colors.white):
    """ Billboard text labels.

    Label that always faces the camera.

    Parameters
    ----------
    pos : array_like, shape (3, )
        Label position in space.
    text : str
        Text to be displayed.
    size : int, optional
        Font size.
    opacity : float, optional
        Label opacity.
    color : array_like, shape (3, ), optional
        Label color.

    Returns
    -------
    Prop
        The created :class:`Prop` instance.
    """
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(pos[0], pos[1], pos[2])
    actor.GetTextProperty().SetFontSize(size)
    actor.GetTextProperty().SetColor(color)
    actor.GetTextProperty().SetJustificationToCentered()
    actor.GetTextProperty().SetShadow(True)
    actor.GetTextProperty().SetBold(True)
    actor.GetTextProperty().SetOpacity(opacity)

    add(actor)
    return Prop(actor)


def _commands():
    """ Toggle splash screen.
    """
    global _renderer, _splash

    message = ('----- Viewer commands -----\n' +
                ' q|e   close window\n' +
                '   r   reset camera\n' +
                '   h   toggle this message\n' +
                ' tab   toggle overlay')

    if _splash is None:
        ren = _renderer
        _splash = canvas(layer=1, interactive=False)

        display(message, y=0.9, shadow=True)

        _renderer = ren
    else:
        _renwin.RemoveRenderer(_splash)
        _splash = None


def show(width=1200, height=600, title=None, info=False, shadows=False, *,
         lmbdown=None, lmbup=None, rmbdown=None, rmbup=None, keydown=None,
         keyup=None, mousemove=None):
    """ Start the VTK event loop.

    Open window for rendering and start the VTK event loop.

    Parameters
    ----------
    width : int, optional
        Window width in pixels.
    height : int, optional
        Window height in pixels.
    title : str, optional
        Window title.
    info : bool, optional
        Show rendering backend information.
    shadows : bool, optional
        Render shadows, experimental.


    This is a blocking function, a script will not advance beyond it until
    the event loop stops. Interaction with displayed objects has to be
    triggered by mouse and keyboard events and corresponding event handlers,
    see below.

    Keyword Arguments
    -----------------
    lmbdown : list[callable]
        Left button press callbacks.
    lmbup : list[callable]
        Left button release callbacks.
    rmbdown : list[callable]
        Right button press callbacks.
    rmbup : list[callable]
        Right button release callbacks.
    keydown : list[callable]
        Key press callbacks.
    keyup : list[callable]
        Key release callbacks.
    mousemove : list[callable]
        Mouse move callbacks.


    .. rubric:: Mouse and keyboard events

    Button and keyboard press and release events as well as mouse move
    events are recognized. Assign a list of callbacks to the corresponding
    keyword argument to register callback functions.

    Note
    ----
    If more than one callback is registered to an event, callbacks are
    executed in the given order.


    .. rubric:: Callback functions

    A callback function's signature has to be defined in the following way:

        .. py:function:: callback(iren, x, y, **kwargs)

           :param iren: Render window interactor.
           :type iren: vtkRenderWindowInteractor
           :param x: Display coordinates of the mouse cursor.
           :type x: int
           :param y: Display coordinates of the mouse cursor.
           :type y: int


    When activated the callback receives a handle to the affected render
    window via the corresponding render window interactor `iren` as well
    as the mouse cursor position inside this windows in pixels.

    The `iren` argument can also be used to query the status of modifier
    keys via its :meth:`GetShiftKey()`, :meth:`GetAltKey()`, and
    :meth:`GetControlKey()` methods.

    Note
    ----
    An object itself can be used as callback when it implements the
    :meth:`__call__` method.


    .. rubric:: Example

    Display a cube. Pressing the space bar will make a screenshot of the
    current window. Since we do not care about the cursor position when
    the space bar was pressed the positional arguments `x` and `y` are
    collected in the `*args` argument and ignored.

    .. code-block:: python
       :linenos:

       import m3sh.vis as vis

       def key_press_callback(iren, *args, **kwargs):
           if iren.GetKeySym().lower() == 'space':
               vis.screenshot('box.png')

       vis.box([-1, -1, -1], [1, 1, 1], edge=True)
       vis.show(title='box', keydown=[key_press_callback])
    """
    # Global state variables that are modified in this function. Should
    # be reset when show() terminates.
    global _renderer, _renwin, _splash

    # This method should only be called once. It can be called again in
    # a script if the active window has been closed.
    if _renwin is not None:
        return

    # Create a window, set its size and title. Multisampling is turned
    # off because of transparent objects.
    _renwin = vtk.vtkRenderWindow()
    _renwin.SetSize(width, height)
    _renwin.SetWindowName(str(title))
    _renwin.SetMultiSamples(0)
    _renwin.SetAlphaBitPlanes(1)
    _renwin.SetNumberOfLayers(2)

    if _renderer is None:
        # Create default viewport. The canvas() function adds the new
        # viewport to the renderers of the _renwin render window.
        canvas()

    lightkit = vtk.vtkLightKit()
    lightkit.AddLightsToRenderer(_renderer)

    # shadows = vtk.vtkShadowMapPass()
    # passes = vtk.vtkRenderPassCollection()
    # passes.AddItem(shadows.GetShadowMapBakerPass())
    # passes.AddItem(shadows)

    # seq = vtk.vtkSequencePass()
    # seq.SetPasses(passes)

    # camera_pass = vtk.vtkCameraPass()
    # camera_pass.SetDelegatePass(seq)

    # _renderer.SetPass(camera_pass)

    # Attach renderers to window. Each renderer is responsible for a
    # viewport inside the main render window.
    while _renderers:
        renderer = _renderers.pop()

        if not _renwin.HasRenderer(renderer):
            _renwin.AddRenderer(renderer)

        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()

        # Changing the render pipeline to include shadows (see above)
        # can also be obtained by this command. Note that shadows are
        # wrong when rendering to non-square windows (known bug).
        if shadows:
            renderer.SetUseShadows(True)

            # Set fixed aspect ratio 1:1 to circumvent the shadow bug.
            size = min(width, height)
            _renwin.SetSize(size, size)

    # Set up the render window interactor with its customized trackball
    # interactor style.
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(_renwin)
    iren.SetInteractorStyle(_MouseInteractorStyle(lmbdown, lmbup,
                                                  rmbdown, rmbup,
                                                  keydown, keyup, mousemove))

    # Add to the list of all render window interactors. If no windows
    # are created by window(), this will be a 1-element list.
    _interactors.append(iren)

    # Calling this method later does not place the axes_widget at the
    # correct spot.
    _renwin.Render()

    # Set application window taskbar and dock icon.
    _app_icon(_renwin)

    # Camera orientation widget was introduced in VTK 9.1, alterantively
    # we can show an axis widget.
    if (vtk.vtkVersion.GetVTKMajorVersion() > 8
            and vtk.vtkVersion.GetVTKMinorVersion() > 0):
        # A reference to the widget has to be maintained to prevent it
        # from being garbage collected immediately.
        iren.axes_widget = vtk.vtkCameraOrientationWidget()
        iren.axes_widget.SetParentRenderer(_renderer)
        iren.axes_widget.GetRepresentation().SetPadding(40, 40)
        iren.axes_widget.SquareResize()
        iren.axes_widget.SetEnabled(True)
    else:
        # Axes display in the lower left corner of each viewport. Only
        # shown in the active viewport, i.e., the one with mouse focus.
        iren.axes_widget = vtk.vtkOrientationMarkerWidget()
        iren.axes_widget.SetOrientationMarker(vtk.vtkAxesActor())
        iren.axes_widget.SetCurrentRenderer(_renderer)
        iren.axes_widget.SetInteractor(iren)
        iren.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        # iren.axes_widget.SetSizeConstraintDimensionSizes(128, 256)
        # iren.axes_widget.SetShouldConstrainSize(True)
        iren.axes_widget.SetInteractive(False)
        iren.axes_widget.SetEnabled(True)

    # Details about OpenGL support and hardware acceleration. Printed on
    # the overlay viewport.
    if info:
        ren = _renderer

        if _splash is None:
            _splash = canvas(layer=1, interactive=False)
        else:
            canvas(_splash)

        display(f"VTK Version {vtk.vtkVersion.GetVTKVersion()}\n" +
                f"OpenGL support {_renwin.SupportsOpenGL()}\n" +
                f"Hardware acceleration {_renwin.IsDirect()}",
                x=0.8, y=.15, shadow=True)

        _renderer = ren

    # Start the window interactor event loop. Closing the windows stops
    # the loop. Alternatively a more expensive polling loop can be used.
    iren.Start()

    # This is helpful in an interactive sessions in IPython. Also allows
    # us to call show() more than once in a script.
    _interactors.clear()
    _renderers.clear()

    _renwin = None
    _renderer = None
    _splash = None


def _show(width=1200, height=600, title=None, position=(0, 0), *, info=False,
         lmbdown=None, lmbup=None, rmbdown=None, rmbup=None, keydown=None,
         keyup=None, mousemove=None):
    """ Start the VTK event loop.

    Open window for rendering and start the VTK event loop.

    Parameters
    ----------
    width : int, optional
        Window width in pixels.
    height : int, optional
        Window height in pixels.
    title : str, optional
        Window title.
    position : 2-tuple (int, int)
        Window position.

    Keyword Arguments
    -----------------
    lmbdown : list[callable]
        Left button press callbacks.
    lmbup : list[callable]
        Left button release callbacks.
    rmbdown : list[callable]
        Right button press callbacks.
    rmbup : list[callable]
        Right button release callbacks.
    keydown : list[callable]
        Key press callbacks.
    keyup : list[callable]
        Key release callbacks.
    mousemove : list[callable]
        Mouse move callbacks.

    Note
    ----
    This is a blocking function, a script will not advance beyond it until
    the event loop stops. Interaction with displayed objects has to be
    triggered by mouse and keyboard events and corresponding event handlers,
    see below.


    .. rubric:: Mouse and keyboard events

    Button press and release events as well as mouse move events are
    recognized. Key press and release events are recognized. Assign a list
    of callbacks to the corresponding keyword argument.

    Note
    ----
    If more than one callback is registered to an event, callbacks are
    executed in the given order.


    .. rubric:: Callback functions

    A callback function's signature has to be defined in the following way:

        .. py:function:: callback(iren, x, y, **kwargs)

           :param iren: Identifies the render window.
           :type iren: vtkRenderWindowInteractor
           :param x: Display coordinates of the mouse cursor.
           :type x: int
           :param y: Display coordinates of the mouse cursor.
           :type y: int


    When activated the callback receives a handle to the affected render
    window via the corresponding render window interactor `iren` as well
    as the mouse cursor position inside this windows in pixels.

    The `iren` argument can also be used to query the status of modifier
    keys via its :meth:`GetShiftKey()`, :meth:`GetAltKey()`, and
    :meth:`GetControlKey()` methods.

    Note
    ----
    An object itself can be used as callback when it implements the
    :meth:`__call__` method.
    """
    # Global state variables that are modified in this function. Should
    # be reset when show() terminates.
    global _renderer, _renwin, _splash

    # This method should only be called once. It can be called again in
    # a script if the active window has been closed.
    if _interactors:
        return

    # Check if a previous window instance has been closed and should now
    # be displayed again.
    if _renwin is not None:
        width, height = _renwin.GetSize()
        title = _renwin.GetWindowName()
        position = _renwin.GetPosition()
    else:
        # Only reset the camera if this is the first time the render
        # window is displayed.
        for renderer in _renderers:
            renderer.ResetCamera()

    # Create a window, set its size and title. Multisampling is turned
    # off because of transparent objects.
    _renwin = vtk.vtkRenderWindow()
    _renwin.SetPosition(position)
    _renwin.SetSize(width, height)
    _renwin.SetWindowName(str(title))
    _renwin.SetMultiSamples(0)
    _renwin.SetAlphaBitPlanes(1)
    _renwin.SetNumberOfLayers(2)

    if _renderer is None:
        # Create default viewport. The canvas() function adds the new
        # viewport to the renderers of the _renwin render window.
        canvas()

    # Attach renderers to window. Each renderer is responsible for a
    # viewport inside the main render window.
    for renderer in _renderers:
        if not _renwin.HasRenderer(renderer):
            _renwin.AddRenderer(renderer)

    # Set up the render window interactor with its customized trackball
    # interactor style.
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(_renwin)
    iren.SetInteractorStyle(_MouseInteractorStyle(lmbdown, lmbup,
                                                  rmbdown, rmbup,
                                                  keydown, keyup, mousemove))

    # Add to the list of all render window interactors. If no windows
    # are created by window(), this will be a 1-element list.
    _interactors.append(iren)

    # Calling this method later does not place the axes_widget at the
    # correct spot.
    _renwin.Render()

    # Set application window taskbar and dock icon.
    _app_icon(_renwin)

    # Camera orientation widget was introduced in VTK 9.1, alterantively
    # we can show an axis widget.
    if (vtk.vtkVersion.GetVTKMajorVersion() > 8
            and vtk.vtkVersion.GetVTKMinorVersion() > 0):
        # A reference to the widget has to be maintained to prevent it
        # from being garbage collected immediately.
        iren.axes_widget = vtk.vtkCameraOrientationWidget()
        iren.axes_widget.SetParentRenderer(_renderer)
        iren.axes_widget.GetRepresentation().SetPadding(40, 40)
        iren.axes_widget.SquareResize()
        iren.axes_widget.SetEnabled(True)
    else:
        # Axes display in the lower left corner of each viewport. Only
        # shown in the active viewport, i.e., the one with mouse focus.
        iren.axes_widget = vtk.vtkOrientationMarkerWidget()
        iren.axes_widget.SetOrientationMarker(vtk.vtkAxesActor())
        iren.axes_widget.SetCurrentRenderer(_renderer)
        iren.axes_widget.SetInteractor(iren)
        iren.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        # iren.axes_widget.SetSizeConstraintDimensionSizes(128, 256)
        # iren.axes_widget.SetShouldConstrainSize(True)
        iren.axes_widget.SetInteractive(False)
        iren.axes_widget.SetEnabled(True)

    # Details about OpenGL support and hardware acceleration. Printed on
    # the overlay viewport.
    if info:
        ren = _renderer

        if _splash is None:
            _splash = canvas(layer=1, interactive=False)
        else:
            canvas(_splash)

        display(f"VTK Version {vtk.vtkVersion.GetVTKVersion()}\n" +
                f"OpenGL support {_renwin.SupportsOpenGL()}\n" +
                f"Hardware acceleration {_renwin.IsDirect()}",
                x=0.8, y=.15, shadow=True)

        _renderer = ren

    # Start the window interactor event loop. Closing the windows stops
    # the loop. Alternatively a more expensive polling loop can be used.
    iren.Start()

    # This is helpful in an interactive sessions in IPython. Also allows
    # us to call show() more than once in a script.
    _interactors.clear()
    # _renderers.clear()

    # _renwin = None
    # _renderer = None
    # _splash = None


def pick(x, y, type='cell', iren=None):
    """ Perform pick action.

    Performs a pick operation at certain display coordinates. Cell and point
    picking is supported.

    Parameters
    ----------
    x : int
        Pick position in display coordinates.
    y : int
        Pick position in display coordinates.
    type : str, optional
        Either 'cell' or 'point'.

    Returns
    -------
    actor : vtkActor
        Results in :obj:`None` if nothing was picked.
    cell_id : int
        Cell identifier, -1 if no cell was picked.
    point_id : int
        Point identifier, -1 if no point was picked.
    point : ndarray
        World coordinates of the picked point.


    By default all render objects created by functions in this module are
    not pickable. To make an actor available for picking, modify its
    :attr:`~Actor.pickable` attribute.

    A successful pick operation returns the picked actor and information about
    the picked cell or point, respectively.

    When picking cells the coordinates of the intersection of the pick ray
    and the picked cell is returned in `point`. In addition to the index
    `cell_id` of the picked cell, the index of the closest vertex of the
    picked cells to this location is returned as `point_id`.

    Note
    ----
    This function returns three values when picking points and four values
    when picking cells.
    """
    # Get the viewport that corresponds to the given location. What happens
    # if window coordinates are out of bounds?
    if iren is None:
        iren = _renwin.GetInteractor()

    ren = iren.FindPokedRenderer(x, y)

    # The pick was successful, i.e., an actor was intersected with the pick
    # ray if the actor returend by the picker is not None.
    if type == 'point':
        picker = vtk.vtkPointPicker()
        picker.Pick(x, y, 0, ren)

        return (picker.GetActor(),
                picker.GetPointId(), np.array(picker.GetPickPosition()))
    elif type == 'cell':
        picker = vtk.vtkCellPicker()
        picker.Pick(x, y, 0, ren)

        return (picker.GetActor(), picker.GetCellId(),
                picker.GetPointId(), np.array(picker.GetPickPosition()))
    else:
        raise ValueError(f"invalid pick style '{type}'")


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='*', type=str, help='OBJ input file')
    parser.add_argument('--bounds', action='store_true', help='show AABB')
    parser.add_argument('--edges', action='store_true', help='show edges')
    parser.add_argument('--flat', action='store_true', help='flat shading')
    # parser.add_argument('--silhouette', action='store_true')

    args = parser.parse_args()
    canvas(color2=colors.black)

    for file in args.file:
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file)
        reader.Update()

        add(mesh := PolyMesh(reader.GetOutput()))

        # Only for testing, remove later...
        # scalars = mesh.points[:, 2]
        # mesh.colorize(scalars, items='verts', interpolate_scalars=True)
        # mesh.lookuptable((min(scalars), max(scalars)), gradient='spectral')

        if mesh.normals is not None:
            quiver(mesh, None, mesh._avg_edge_length()[0])

        if args.bounds:
            aabb(mesh)

        if args.flat:
            mesh.prop.GetProperty().SetInterpolationToFlat()

        if args.edges:
            mesh.edges(width=1, color=colors.ivory_black)

        # if args.silhouette:
        #     mesh.silhouette(width=4, color=colors.black)

    # Window title holds the list of all given filenames even if only
    # one file was given.
    show(title=args.file, info=True)


def _show_vertex_index_cb(iren, x, y, **kwargs):
    """
    """
    print(pick(x, y, type='cell', iren=iren)[1:])


def _show_cell_labels(obj):
    """
    """
    ids = vtk.vtkIdFilter()
    ids.SetInputData(obj._vtk_polydata)
    ids.SetPointIds(False)
    ids.SetCellIds(True)

    cc = vtk.vtkCellCenters()
    cc.SetInputConnection(ids.GetOutputPort())

    # Labels are drawn on top of all other objects. It should be possible
    # to take visibility into account.
    mapper = vtk.vtkLabeledDataMapper()
    mapper.SetInputConnection(cc.GetOutputPort())
    mapper.SetLabelModeToLabelScalars()

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)

    add(actor)
    return Prop(actor)


class Prop:
    """ Render object.

    Base class for all VTK wrappers.

    Parameters
    ----------
    prop : vtkProp
        Instance of a render object.

    Note
    ----
    Wrapped objects are not pickable by default.
    """

    def __init__(self, prop):
        self._vtk_prop = prop
        self._vtk_prop.SetPickable(False)

    @property
    def prop(self):
        """ Wrapped VTK instance.

        Access the wrapped render object. Exposes all low-level
        interaction with this object.

        :type: vtkProp
        """
        return self._vtk_prop

    @property
    def actor(self):
        """ Actor instance.

        Access the associated actor. Generically this is equal to
        the :attr:`prop` attribute.

        :type: vtkActor

        Note
        ----
        Derived classes should provide their own implementation.
        """
        return self._vtk_prop

    @property
    def name(self):
        """ Prop name.

        Set and get name of the prop (empty by default).

        :type: str
        """
        return self._vtk_prop.GetObjectName()

    @name.setter
    def name(self, value):
        self._vtk_prop.SetObjectName(str(value))

    @property
    def bounds(self):
        """ Axis aligned bounding box.

        Pair holding the min and max values along each of the three
        coordinate dimensions.

        :type: tuple[ndarray, ndarray]
        """
        bounds = self._vtk_prop.GetBounds()
        return np.array(bounds[0::2]), np.array(bounds[1::2])

    @property
    def visible(self):
        """ Visibility property.

        Query and toggle object visibility.

        :type: bool
        """
        return self._vtk_prop.GetVisibility()

    @visible.setter
    def visible(self, value):
        self._vtk_prop.SetVisibility(bool(value))

    @property
    def pickable(self):
        """ Pickable property.

        Query and toggle whether object geometry can be picked. Objects
        are not pickable by default.

        :type: bool
        """
        return self._vtk_prop.GetPickable()

    @pickable.setter
    def pickable(self, value):
        self._vtk_prop.SetPickable(bool(value))


class PropertyMixin:
    # Assumes that self._vtk_prop is derived from a class that provides
    # the GetProperty() method.

    @property
    def opacity(self):
        """ Opacity property.

        Set and get opacity value.

        :type: float
        """
        return self._vtk_prop.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, value):
        self._vtk_prop.GetProperty().SetOpacity(value)

    # Render and interpolation styles... needs more work!
    # @property
    # def flat(self):
    #     return self._vtk_prop.GetProperty().GetOpacity()

    # @opacity.setter
    # def flat(self, value):
    #     self._vtk_prop.GetProperty().SetOpacity(value)


class MapperMixin:
    # Assumes that self._vtk_prop is derived from a class that provides
    # the GetMapper() method.

    def lookuptable(self, range=None, gradient=None, logscale=None,
                    size=None, **kwargs):
        """ Modify lookup table properties.

        An objects' lookup tables determines how entries of the scalar array
        are translated to color values. Smooth color gradients are defined
        by the color schemes 'hot', 'cool', 'jet', and 'grey'. The color
        schemes 'spectral', 'diverging', 'blue', 'orange', and 'purple'
        define a discrete color series.

        Parameters
        ----------
        range : (float, float)
            Accpeted range of scalar values, default [0, 1].
        gradient : str
            Color scheme identifier.
        logscale : bool
            Switch between linear and logarithmic scale.
        size : int
            Size of lookup table.

        Note
        ----
        Lookup tables have no effect when directly mapping RGB color values.


        If the corresponding keyword arguments are provied, out of range
        values are marked with the `below`, `above`, and `nan` colors. Note
        that those colors also have an alpha value to control opacity.

        Keyword arguments
        -----------------
        below : array_like, shape (4, )
            Color for scalars below range.
        above: array_like, shape (4, )
            Color for scalars above range.
        nan : array_like, shape (4, )
            Special color for NaN scalar values.

        Note
        ----
        Any arguments not provided have no affect on the corresponding lookup
        table property.
        """
        # The current lookup table. Properties are modified according to the
        # given parameters. None values preserve the corresponding property.
        lut = self._vtk_prop.GetMapper().GetLookupTable()
        _tweak_lut(lut, range, gradient, logscale, size, **kwargs)


class GlyphMixin:

    @property
    def size(self):
        """ Size property.

        Set and get global size for all glyphs.

        :type: float
        """
        # Global scale factor that is multiplied by scalars values if
        # used to set per point scale factors.
        return self._vtk_glyph.GetScaleFactor()

    @size.setter
    def size(self, value):
        self._vtk_glyph.SetScaleFactor(value)

    def scale(self, values):
        """ Scale glyph geometry.

        Parameters
        ----------
        values : array_like
            Per point scale factors.

        Note
        ----
        Scale factors and :attr:`size` are multiplied to obtain the final
        size of a glyph.
        """
        self._scalars = np.asarray(values)
        self._vtk_polydata.GetPointData().SetScalars(numpy_to_vtk(self._scalars))
        self._vtk_glyph.SetScaling(True)
        self._vtk_glyph.SetScaleModeToScaleByScalar()

    @property
    def color(self):
        """ Color property.

        Setting the color attribute disables coloring using previously
        set scalars with :meth:`colorize`.

        :type: array_like, shape (3, )
        """
        return self._vtk_prop.GetProperty().GetColor()

    @color.setter
    def color(self, value):
        self._vtk_prop.GetMapper().SetScalarVisibility(False)
        self._vtk_prop.GetProperty().SetColor(value)

    def colorize(self, scalars):
        numpts = self._vtk_polydata.GetNumberOfPoints()
        shape = np.shape(scalars)

        if shape == (numpts, ) or shape == (numpts, 3):
            self._colors = np.asarray(scalars)

            colors = numpy_to_vtk(self._colors)
            colors.SetName('color')

            # This will replaces an array of the same name if present.
            self._vtk_polydata.GetPointData().AddArray(colors)

            mapper = self._vtk_prop.GetMapper()
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray('color')
            mapper.SetScalarVisibility(True)

            # Direct colors is the array has 3 components, otherwise
            # map the scalars through the lookup table.
            if shape == (numpts, ):
                mapper.SetColorModeToMapScalars()
            else:
                mapper.SetColorModeToDirectScalars()

    def modified(self):
        """ Update notification.

        Should be called when the contents of a shared data buffer are
        modified.

        Note
        ----
        Changing the **shape** of a shared data buffer is likely to
        result in a segmentation fault or other undefined behavior.
        """
        self._vtk_polydata.GetPoints().Modified()
        self._vtk_polydata.GetPointData().Modified()
        self._vtk_polydata.GetCellData().Modified()


# class BoxArray(Actor):

#     def __init__(self, boxes):
#         prop = vtk.vtkPropAssembly()

#         for box in boxes:
#             prop.AddPart(self._build(box[0, :], box[1, :]))

#         super().__init__(prop)

#     def _build(self, a, b):
#         cube = vtk.vtkCubeSource()
#         cube.SetBounds(a[0], b[0],
#                        a[1], b[1],
#                        a[2], b[2])
#         cube.Update()

#         mapper = vtk.vtkPolyDataMapper()
#         mapper.SetInputData(cube.GetOutput())

#         actor = vtk.vtkActor()
#         actor.SetMapper(mapper)

#         return actor


class _ConeField():
    """ Family of cones.

    Visual representation of a family of (double) cones. Individual cones
    are defined by their apex, height, and base radius. Either the height
    or radius may be replaced by an angle. Specifiying all three values is
    considered as invalid input.

    Parameters
    ----------
    points : array_like
        Point cloud in 3-space, one point per row. Each point defines
        the apex of a cone.
    vectors : array_like
        Vector field in 3-space, one vector per row. Vectors define
        cone axis directions and don't have to be normalized.
    angle : array_like
        One angle per cone. Angles are measured in radians.
    radius : array_like
        One positive radius per cone.
    height : array_like
        One non-negative height per cone.
    double : bool
        :obj:`True` will generate double cones.
    cap : bool
        :obj:`True` to close the cone along its base.
    resolution : int
        The number of vertices along the base circle.

    Note
    ----
    One of `angle`, `height`, `radius` has to be explicitly set the
    :obj:`None` during initialization. Note that `angle` refers to half the
    opening angle of a cone.
    """

    def __init__(self, points, vectors, angle, radius, height, double,
                 cap, resolution):
        # Convert to ndarray. The data buffers of the resulting arrays
        # are then shared with VTK. If the arguments are already of type
        # ndarray no data duplication happens.
        self.points = np.asarray(points)
        self.vecpos = np.asarray(vectors)

        # This will be shared by both polydata objects when displaying
        # double cones.
        self._points = vtk.vtkPoints()
        self._points.SetData(numpy_to_vtk(self.points))

        # The index array is also only needed once and shared in case of
        # double cones.
        indices = vtk.vtkIntArray()
        indices.SetNumberOfComponents(1)
        indices.SetName('index')

        for id in range(len(self.points)):
            indices.InsertNextTuple1(id)

        self._indices = indices

        # We define cones from radius and height. If one of those is missing
        # it is computed using the angle value (required in this case).
        if (radius is None
                and (angle is not None and height is not None)):
            angle = np.atleast_1d(angle)
            height = np.atleast_1d(height)
            radius = height * np.tan(angle)
        elif (height is None
                and (angle is not None and radius is not None)):
            angle = np.atleast_1d(angle)
            radius = np.atleast_1d(radius)
            height = radius / np.tan(angle)
        else:
            raise ValueError('wrong number of arguments')

        if double:
            self.vecneg = -self.vecpos

            prop = vtk.vtkPropAssembly()
            prop.AddPart(self._build(self.vecpos, radius, height,
                                     cap, resolution))
            prop.AddPart(self._build(self.vecneg, radius, height,
                                     cap, resolution))
        else:
            prop = self._build(self.vecpos, radius, height, cap, resolution)

        super().__init__(prop)

    def _build(self, axis, radius, height, capping, resolution):
        """ Build glyph representation.
        """
        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(self._points)
        self._polydata.GetPointData().SetVectors(numpy_to_vtk(axis))
        self._polydata.GetPointData().AddArray(self._indices)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self._polydata)

        # The source shape used for glyphs. If rendering is too slow when
        # there is a large number of glyphs, the resolution can be reduced.
        for id, (h, r) in enumerate(zip(height, radius, strict=True)):
            cone = vtk.vtkConeSource()
            cone.SetResolution(resolution)
            cone.SetCapping(capping)

            # Cone source objects only accepts non-negative values for height
            # and radius, angles have to be in the range [0.0, 90.0]. In all
            # other cases no shape is displayed without any error message.
            cone.SetHeight(h)
            cone.SetRadius(r)
            cone.SetCenter(-0.5 * h, 0.0, 0.0)

            glyph.SetSourceConnection(id, cone.GetOutputPort())

        glyph.SetRange(0, len(self.points) - 1)
        glyph.SetIndexModeToScalar()
        glyph.SetInputArrayToProcess(0, 0, 0, 0, 'index')
        glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        glyph.ScalingOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor


class _VectorField():
    """
    """

    def __init__(self, data, scale=1.0, color=colors.green_pale,
                 shaft_radius=0.025, tip_radius=0.05, tip_length=0.5,
                 resolution=6):
        # The source shape used for glyphs. Reduce resolution if rendering
        # is too slow when a large number of glyphs is displayed.
        arrow = self.arrow_source(shaft_radius, tip_radius, tip_length,
                                  resolution)

        if isinstance(data, vtk.vtkPolyData):
            self._points = vtk_to_numpy(data.GetPoints().GetData())
            self._vectors = vtk_to_numpy(data.GetPointData().GetNormals())

            self._vtk_polydata = data

            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(data)
            glyph.SetSourceConnection(arrow.GetOutputPort())
            glyph.OrientOn()
            glyph.SetVectorModeToUseNormal()
            glyph.SetScaleModeToDataScalingOff()
            glyph.SetScaleFactor(scale)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
        else:
            self._points = np.asarray(data[0])
            self._vectors = np.asarray(data[1])

            if not np.all(np.isfinite(self._vectors)):
                raise ValueError('infinite vector coordinate(s)')

            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(self._points))

            scalars = vtk.vtkFloatArray()
            scalars.SetNumberOfComponents(1)
            scalars.SetName('glyph_scale')

            colors = vtk.vtkFloatArray()
            colors.SetNumberOfComponents(3)
            colors.SetName('glyph_color')

            if np.shape(color) == (3, ):
                color_ = iter(lambda: color, None)
            else:
                color_ = iter(color)

            if isinstance(scale, int) or isinstance(scale, float):
                scale_ = iter(lambda: scale, None)
            else:
                scale_ = iter(scale)

            # Starting with Python 3.10 zip supports the 'strict' keyword
            # argument. Helps to check consistency of points and vectors.
            for _ in zip(self._points, self._vectors, strict=True):
                scalars.InsertNextTuple1(next(scale_))
                colors.InsertNextTuple3(*next(color_))

            self._vtk_polydata = vtk.vtkPolyData()
            self._vtk_polydata.SetPoints(points)

            point_data = self._vtk_polydata.GetPointData()
            point_data.SetVectors(numpy_to_vtk(self._vectors))
            point_data.AddArray(scalars)
            point_data.AddArray(colors)
            point_data.SetActiveScalars('glyph_scale')

            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(self._vtk_polydata)
            glyph.SetSourceConnection(arrow.GetOutputPort())
            glyph.OrientOn()
            glyph.SetVectorModeToUseVector()
            glyph.ScalingOn()
            glyph.SetScaleModeToScaleByScalar()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            mapper.SelectColorArray('glyph_color')

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

        super().__init__(actor)

    @staticmethod
    def arrow_source(shaft_radius, tip_radius, tip_length, resolution):
        """
        """
        arrow = vtk.vtkArrowSource()
        arrow.SetTipRadius(tip_radius)
        arrow.SetTipLength(tip_length)
        arrow.SetTipResolution(resolution)
        arrow.SetShaftRadius(shaft_radius)
        arrow.SetShaftResolution(resolution)

        return arrow

    def modified(self):
        """
        """
        self._vtk_polydata.GetPoints().Modified()
        self._vtk_polydata.GetPointData().Modified()





class OrientedGlyphs(Prop, PropertyMixin, MapperMixin, GlyphMixin):
    """ Glyph base class.

    Base class for all simple oriented glyphs. Displays a scaled and
    oriented `source` shape at all point locations.

    Parameters
    ----------
    points : array_like
        Point coordinates.
    vectors : array_like
        Orientation defining vectors.
    source
        Source object.
    src_xform : vtkTransform
        Source transformation.

    Note
    ----
    If `array_like` parameters are of type :class:`~numpy.ndarray` their
    data buffer is shared with VTK's data objects (use copies to decouple
    storage).
    """

    def __init__(self, points, vectors, source, src_xform):
        if isinstance(points, Prop):
            points = points._vtk_polydata

        if isinstance(points, vtk.vtkPolyData):
            self._points = vtk_to_numpy(points.GetPoints().GetData())

            # Normals should either be stored with the polydata instance
            # (vectors should be None in this case) or specified by the
            # vectors parameter (in this case the polydata should not
            # define normals).
            if ((normals := points.GetPointData().GetNormals()) is not None
                    and vectors is not None):
                raise ValueError('ambiguous normal specification')

            if normals is not None:
                self._vectors = vtk_to_numpy(normals)
            elif vectors is not None:
                self._vector = np.asarray(vectors)
                points.GetPointData().SetNormals(numpy_to_vtk(self._vector))
            else:
                raise ValueError('no normals specified')

            self._vtk_points = points.GetPoints()
            self._vtk_polydata = points
        else:
            self._points = np.asarray(points)
            self._vectors = np.asarray(vectors)

            self._vtk_points = vtk.vtkPoints()
            self._vtk_points.SetData(numpy_to_vtk(self._points))

            self._vtk_polydata = vtk.vtkPolyData()
            self._vtk_polydata.SetPoints(self._vtk_points)

            pointdata = self._vtk_polydata.GetPointData()
            pointdata.SetNormals(numpy_to_vtk(self._vectors))

        self._vtk_source = source

        self._vtk_glyph = vtk.vtkGlyph3D()
        self._vtk_glyph.SetInputData(self._vtk_polydata)
        self._vtk_glyph.SetSourceConnection(source.GetOutputPort())
        self._vtk_glyph.OrientOn()
        self._vtk_glyph.SetVectorModeToUseNormal()

        # This allows scaling by the global scale factor but does not
        # use local scale factors associated with individual points.
        self._vtk_glyph.SetScaleModeToDataScalingOff()

        if src_xform is not None:
            self._vtk_glyph.SetSourceTransform(src_xform)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._vtk_glyph.GetOutputPort())
        mapper.SetLookupTable(_generic_lut(gradient='spectral'))
        mapper.SetUseLookupTableScalarRange(True)
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        super().__init__(actor)


class Spheres(Prop, PropertyMixin, MapperMixin, GlyphMixin):
    """ Point cloud."""

    def __init__(self, points):
        self._points = np.asarray(points)

        self._vtk_points = vtk.vtkPoints()
        self._vtk_points.SetData(numpy_to_vtk(self._points))

        self._vtk_polydata = vtk.vtkPolyData()
        self._vtk_polydata.SetPoints(self._vtk_points)

        self._source = self._sphere()

        self._vtk_glyph = vtk.vtkGlyph3D()
        self._vtk_glyph.SetInputData(self._vtk_polydata)
        self._vtk_glyph.SetSourceConnection(self._source.GetOutputPort())
        self._vtk_glyph.SetScaleModeToScaleByScalar()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._vtk_glyph.GetOutputPort())
        mapper.SetLookupTable(_generic_lut(gradient='spectral'))
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        super().__init__(actor)

    @staticmethod
    def _sphere(resolution=16):
        sphere = vtk.vtkSphereSource()

        sphere.SetPhiResolution(resolution)
        sphere.SetThetaResolution(resolution)

        return sphere

    @staticmethod
    def _cube():
        return vtk.vtkCubeSource()

    @property
    def style(self):
        if isinstance(self._source, vtk.vtkCubeSource):
            return 'cubes'
        elif isinstance(self._source, vtk.vtkSphereSource):
            return 'spheres'

        return 'undefined'

    @style.setter
    def style(self, value):
        if value == 'cubes':
            self._source = self._cube()
        elif value == 'spheres':
            self._source = self._sphere()

        self._vtk_glyph.SetSourceConnection(self._source.GetOutputPort())


class Arrows(OrientedGlyphs):

    def __init__(self, points, vectors, **kwargs):
        super().__init__(points, vectors, *self._arrow(**kwargs))

    @staticmethod
    def _arrow(shaft_radius=0.025, tip_radius=0.05, tip_length=0.5,
               resolution=6):
        arrow = vtk.vtkArrowSource()

        arrow.SetTipRadius(tip_radius)
        arrow.SetTipLength(tip_length)
        arrow.SetTipResolution(resolution)

        arrow.SetShaftRadius(shaft_radius)
        arrow.SetShaftResolution(resolution)

        return arrow, None


class Circles(OrientedGlyphs):

    def __init__(self, points, vectors, radius):
        super().__init__(points, vectors, self._arc())

        self.scale(radius)

    @staticmethod
    def _arc(resolution=60):
        arc = vtk.vtkArcSource()

        arc.SetCenter(0.0, 0.0, 0.0)
        arc.SetNormal(1.0, 0.0, 0.0)
        arc.SetPolarVector(0.0, 1.0, 0.0)
        arc.SetAngle(360.0)
        arc.SetUseNormalAndAngle(True)
        arc.SetResolution(resolution)

        return arc, None

    @property
    def width(self):
        return self._vtk_prop.GetProperty().GetLineWidth()

    @width.setter
    def width(self, value):
        self._vtk_prop.GetProperty().SetLineWidth(value)

    @property
    def radius(self):
        return self._scalars

    @radius.setter
    def radius(self, value):
        self.scale(value)

    @property
    def style(self):
        if self._vtk_prop.GetProperty().GetRenderLinesAsTubes():
            return 'tubes'

        return 'lines'

    @style.setter
    def style(self, value):
        if value == 'tubes':
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(True)
        else:
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(False)


class Cones(Prop, PropertyMixin):

    def __init__(self, points, vectors, angle, radius, height, double,
                 cap, resolution):

        self._points = np.asarray(points)
        self._vectors = np.asarray(vectors)

        self._vtk_points = vtk.vtkPoints()
        self._vtk_points.SetData(numpy_to_vtk(self._points))

        self._vtk_polydata = vtk.vtkPolyData()
        self._vtk_polydata.SetPoints(self._vtk_points)

        pointdata = self._vtk_polydata.GetPointData()
        pointdata.SetVectors(numpy_to_vtk(self._vectors))

        indices = vtk.vtkIntArray()
        indices.SetNumberOfComponents(1)
        indices.SetName('index')

        for id in range(len(self._points)):
            indices.InsertNextTuple1(id)

        self._vtk_polydata.GetPointData().AddArray(indices)

        # colors = vtk.vtkFloatArray()
        # colors.SetNumberOfComponents(3)
        # colors.SetName('color')

        # # for id in range(len(self._points)):
        # #     colors.InsertNextTuple1(id * 0.5)

        # colors.InsertNextTuple3(1.0, 0.0, 0.0)
        # colors.InsertNextTuple3(0.0, 1.0, 0.0)
        # colors.InsertNextTuple3(0.0, 0.0, 1.0)

        # self._vtk_polydata.GetPointData().AddArray(colors)

        if (radius is None
                and (angle is not None and height is not None)):
            angle = np.atleast_1d(angle)
            height = np.atleast_1d(height)
            radius = height * np.tan(angle)
        elif (height is None
                and (angle is not None and radius is not None)):
            angle = np.atleast_1d(angle)
            radius = np.atleast_1d(radius)
            height = radius / np.tan(angle)

        prop = vtk.vtkPropAssembly()
        prop.AddPart(self._cones(height, radius, False, cap, resolution))

        if double:
            prop.AddPart(self._cones(height, radius, True, cap, resolution))

        super().__init__(prop)

    @property
    def actor(self):
        """ Actor instance.

        Actor associated to the first part of the assembly.

        :type: vtkActor
        """
        parts = self._vtk_prop.GetParts()
        parts.InitTraversal()

        return parts.GetNextProp()

    @property
    def color(self):
        """ Color property.

        Setting the global color attribute disables coloring using previously
        set scalars with :meth:`colorize`.

        :type: array_like, shape (3, )
        """
        return self._vtk_prop.GetProperty().GetColor()

    @color.setter
    def color(self, value):
        for actor in self._vtk_prop.GetParts():
            actor.GetProperty().SetColor(value)
            actor.GetMapper().SetScalarVisibility(False)

    def colorize(self, scalars):
        numpts = self._vtk_polydata.GetNumberOfPoints()
        shape = np.shape(scalars)

        if shape == (numpts, ) or shape == (numpts, 3):
            self._colors = np.asarray(scalars)

            colors = numpy_to_vtk(self._colors)
            colors.SetName('color')

            # This will replaces an array of the same name if present.
            self._vtk_polydata.GetPointData().AddArray(colors)

            for actor in self._vtk_prop.GetParts():
                mapper = actor.GetMapper()
                mapper.SetScalarVisibility(True)

                # Direct colors is the array has 3 components, otherwise
                # map the scalars through the lookup table.
                if shape == (numpts, ):
                    mapper.SetColorModeToMapScalars()
                else:
                    mapper.SetColorModeToDirectScalars()

    def lookuptable(self, range=None, gradient=None, logscale=None,
                    size=None, **kwargs):
        """ Modify lookup table properties.

        An objects lookup tables determines how entries of the scalar array
        are translated to color values. Lookup tables have no effect when
        directly mapping RGB color values.

        Parameters
        ----------
        range : (float, float)
            Accpeted range of scalar values.
        gradient : str
            Color scheme identifier, see below.
        logscale : bool
            Switch between linear and logarithmic scale.
        size : int
            Size of lookup table.

        Keyword arguments
        -----------------
        below : array_like, shape (4, )
            Color for scalars below the specified range.
        above: array_like, shape (4, )
            Color for scalars above the specified range.
        nan : array_like, shape (4, )
            Special color for NaN scalar values.


        Smooth color gradients are defined by the color scheme identifiers
        'hot', 'cool', 'jet', and 'grey'. The color schemes 'spectral',
        'diverging', 'blue', 'orange', and 'purple' define a discrete color
        series.

        If provied, out of range values are marked with the `below`,
        `above`, and `nan` colors. Not that those colors also have an alpha
        intensity value to control opacity.

        Note
        ----
        Arguments not provided have no affect on the corresponding lookup
        table property.
        """
        for actor in self._vtk_prop.GetParts():
            lut = actor.GetMapper().GetLookupTable()
            _tweak_lut(lut, range, gradient, logscale, size, **kwargs)

    @staticmethod
    def _cone(height, radius, capping=False, resolution=24):
        cone = vtk.vtkConeSource()

        cone.SetHeight(height)
        cone.SetRadius(radius)
        cone.SetCenter(-0.5 * height, 0.0, 0.0)

        cone.SetResolution(resolution)
        cone.SetCapping(capping)

        return cone

    def _cones(self, height, radius, flip, capping, resolution):
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self._vtk_polydata)

        for i, (h, r) in enumerate(zip(height, radius, strict=True)):
            cone = self._cone(h, r, capping, resolution)
            glyph.SetSourceConnection(i, cone.GetOutputPort())

        if flip:
            xform = vtk.vtkTransform()
            xform.Identity()
            xform.RotateY(180.0)

            glyph.SetSourceTransform(xform)

        glyph.SetRange(0, len(self._points) - 1)
        glyph.SetIndexModeToScalar()
        glyph.SetInputArrayToProcess(0, 0, 0, 0, 'index')

        glyph.SetColorModeToColorByScalar()
        glyph.SetInputArrayToProcess(3, 0, 0, 0, 'color')

        glyph.ScalingOff()

        glyph.OrientOn()
        glyph.SetVectorModeToUseVector()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(_generic_lut(gradient='spectral'))
        mapper.SetUseLookupTableScalarRange(True)
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor


class Disks(OrientedGlyphs):

    def __init__(self, points, vectors):
        super().__init__(points, vectors, *self._disk())

    @staticmethod
    def _disk(resolution=60):
        disk = vtk.vtkDiskSource()

        disk.SetInnerRadius(0.0)
        disk.SetOuterRadius(1.0)
        disk.SetCircumferentialResolution(resolution)

        xform = vtk.vtkTransform()
        xform.Identity()
        xform.RotateY(90.0)

        return disk, xform


class __VectorField():

    def __init__(self, points, vectors, scale=1.0, color=(0.5, 0.5, 0.5),
                 shaft_radius=0.025, tip_radius=0.05, tip_length=0.5,
                 resolution=6):
        # Prepare the data buffers used by VTK and fill them with the data
        # provided.
        self._points = vtk.vtkPoints()
        self._points.SetData(numpy_to_vtk(points))

        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName('glyph_scale')

        colors = vtk.vtkFloatArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('glyph_color')

        if np.shape(color) == (3, ):
            color_ = iter(lambda: color, None)
        else:
            color_ = iter(color)

        if isinstance(scale, int) or isinstance(scale, float):
            scale_ = iter(lambda: scale, None)
        else:
            scale_ = iter(scale)

        # Starting with Python 3.10 zip supports the 'strict' keyword argument.
        # Helps to check consistency of points and vectors argument.
        for _, vec in zip(points, vectors, strict=True):
            if np.all(np.isfinite(vec)):
                scalars.InsertNextTuple1(next(scale_))
                colors.InsertNextTuple3(*next(color_))

        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(self._points)
        self._polydata.GetPointData().SetVectors(numpy_to_vtk(vectors))
        self._polydata.GetPointData().AddArray(scalars)
        self._polydata.GetPointData().AddArray(colors)
        self._polydata.GetPointData().SetActiveScalars('glyph_scale')

        # The source shape used for glyphs. If rendering is too slow when there
        # is a large number of glyphs, the resolution of each can be reduced.
        arrow = vtk.vtkArrowSource()
        arrow.SetTipRadius(tip_radius)
        arrow.SetTipLength(tip_length)
        arrow.SetTipResolution(resolution)
        arrow.SetShaftRadius(shaft_radius)
        arrow.SetShaftResolution(resolution)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self._polydata)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        glyph.ScalingOn()
        glyph.SetScaleModeToScaleByScalar()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        mapper.SelectColorArray('glyph_color')

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        super().__init__(actor)

    def modified(self):
        self._polydata.GetPoints().Modified()
        self._polydata.GetPointData().Modified()


class _PolyData():
    def __init__(self, data, *, verts=None, lines=None, faces=None):
        if isinstance(data, vtk.vtkPolyData):
            self._points = vtk_to_numpy(data.GetPoints().GetData())
            self._vtk_points = data.GetPoints()
            self._vtk_polydata = data
        else:
            self._points = np.asarray(data)

            self._vtk_points = vtk.vtkPoints()
            self._vtk_points.SetData(numpy_to_vtk(self._points))

            self._vtk_polydata = vtk.vtkPolyData()
            self._vtk_polydata.SetPoints(self._vtk_points)

            if verts is not None:
                cells = vtk.vtkCellArray()
                for v in verts:
                    cells.InsertNextCell(1, [int(v)])
                self._vtk_polydata.SetVerts(cells)

            # if lines is not None:
            #     cells = vtk.vtkCellArray()
            #     for i in range(len(edges) - 1):
            #         if lines[i] > -1 and lines[i+1] > -1:
            #             line = vtk.vtkIdList()
            #             line.InsertNextId(lines[i])
            #             line.InsertNextId(lines[i+1])
            #             cells.InsertNextCell(line)
            #     self._vtk_polydata.SetLines(cells)

            if faces is not None:
                cells = vtk.vtkCellArray()
                for f in faces:
                    face = vtk.vtkIdList()
                    for v in f:
                        face.InsertNextId(int(v))
                    cells.InsertNextCell(face)
                self._vtk_polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_polydata)
        mapper.SetLookupTable(_generic_lut())
        mapper.SetUseLookupTableScalarRange(True)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        super().__init__(actor)

        self._draggable = None
        self._scalars = None

    @property
    def points(self):
        """ Point coordinate array access.

        The contents of this array may be changed in place. Calling
        :meth:`modified` will update the visual representation on the
        next render pass.

        :type: ~numpy.ndarray
        """
        return self._points

    @property
    def normals(self):
        """
        """
        return vtk_to_numpy(self._vtk_polydata.GetPointData().GetNormals())

    def colorize(self, scalars, items, range=None, gradient=None,
                 logscale=None, size=None):
        """ Colorize polygonal data.

        Colorize by assinging vertex colors or face colors. Vertex colors
        are interpolated across faces. Colors can be specified directly as
        RGB intensity triples or via a color map that maps scalar values
        to RGB values.

        Parameters
        ----------
        scalars : ~numpy.ndarray
            Scalar values. Either one scalar per item or one RGB color
            triple per item.
        items : str
            Either 'points' or 'cells'.
        range : (float, float), optional
            Lookup table range. Defaults to the range given by the
            smallest and largest scalar value.
        gradient : str, optional
            Color scheme identifier.
        logscale : bool, optional
            Toggle logarithmic scaling.
        size : int, optional
            Size of lookup table.


        Mapping scalars to colors uses a lookup table managed by the
        :attr:`mapper` instance of an actor. The `range`, `gradient`, and
        `logscale` arguments directly influence the lookup table. Lookup
        tables can be further customized via the :meth:`lookuptable` method.

        Use the :func:`colorbar` function to display a visual representation
        of a lookup table.

        Note
        ----
        When specifying colors directly by RGB triples, all arguments
        except `scalars` and `items` are ignored.
        """
        if scalars is not None:
            if items == 'points':
                self._set_point_scalars(scalars)
            elif items == 'cells':
                self._set_cell_scalars(scalars)
            else:
                raise ValueError(f"invalid item argument '{items}'")

            self.lookuptable(range, gradient, logscale, size)
        else:
            self._reset_scalars()

    def edges(self, style=None, width=None, color=None):
        """ Edge display.

        Set visual properties of edges.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes'. :obj:`False` to disable.
        width : int, optional
            Edge width in pixels.
        color : array_like, shape (3, ), optional
            Edge color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        edge display property.
        """
        if style == 'lines':
            self.prop.GetProperty().SetRenderLinesAsTubes(False)
            self.prop.GetProperty().SetEdgeVisibility(True)
        elif style == 'tubes':
            self.prop.GetProperty().SetRenderLinesAsTubes(True)
            self.prop.GetProperty().SetEdgeVisibility(True)
        elif style == '' or style is False:
            self.prop.GetProperty().SetEdgeVisibility(False)
        else:
            self.prop.GetProperty().SetEdgeVisibility(True)

        if width is not None:
            self.prop.GetProperty().SetLineWidth(width)

        if color is not None:
            self.prop.GetProperty().SetEdgeColor(color)

    def silhouette(self, style=None, width=None, color=None):
        """ Silhouette display.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes'. :obj:`False` to disable.
        width : int, optional
            Edge width in pixels.
        color : array_like, shape (3, ), optional
            Edge color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        silhouette display property.
        """
        if not hasattr(self, '_silhouette'):
            self._silhouette = None

        if style == '' or style is False:
            delete(self._silhouette)
            self._silhouette = None
            return

        if self._silhouette is None:
            outline = vtk.vtkPolyDataSilhouette()
            outline.SetInputData(self._vtk_polydata)
            outline.SetCamera(_renderer.GetActiveCamera())
            outline.SetEnableFeatureAngle(False)
            outline.SetBorderEdges(True)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPickable(False)

            if color is None:
                actor.GetProperty().SetColor(colors.black)

            if width is None:
                actor.GetProperty().SetLineWidth(2)

            add(actor)
            self._silhouette = actor
        else:
            actor = self._silhouette

        if width is not None:
            actor.GetProperty().SetLineWidth(width)

        if style == 'lines':
            actor.GetProperty().SetRenderLinesAsTubes(False)
        elif style == 'tubes':
            actor.GetProperty().SetRenderLinesAsTubes(True)

        if color is not None:
            actor.GetProperty().SetColor(color)

    def modified(self):
        self._vtk_points.Modified()
        super().Modified()

    def _avg_edge_length(self):
        filter = vtk.vtkExtractEdges()
        filter.SetInputData(self._vtk_polydata)
        filter.Update()

        points = vtk_to_numpy(filter.GetOutput().GetPoints().GetData())
        edges = filter.GetOutput().GetLines()
        edge_iter = edges.NewIterator()

        min, max, sum = np.inf, 0.0, 0.0

        while not edge_iter.IsDoneWithTraversal():
            edge = edge_iter.GetCurrentCell()
            i, j = edge.GetId(0), edge.GetId(1)

            length = np.linalg.norm(points[i, :] - points[j, :])
            sum += length

            min = length if length < min else min
            max = length if length > max else max

            edge_iter.GoToNextCell()

        return sum / edges.GetNumberOfCells(), min, max

    def _set_cell_scalars(self, value):
        polydata = self._vtk_polydata

        self._set_scalars(polydata.GetCellData(),
                          polydata.GetNumberOfCells(), value)

        polydata.GetPointData().SetScalars(None)
        polydata.GetPointData().Modified()

        self.prop.GetMapper().SetScalarModeToUseCellData()


class PolyData(Prop, PropertyMixin, MapperMixin):
    """ Polygonal shape wrapper.

    Manages visual properties of a polygonal shape. If `data` is an
    `array_like` representation of a point set, it is converted to an
    equivalent :class:`~numpy.ndarray` instance. The created array is
    then accessible via the :attr:`points` attribute.

    Parameters
    ----------
    data : array_like or vtkPolyData
        Point data.
    verts : list, optional
        Combinatorial vertex definitions.
    lines : list, optional
        Combinatorial line definitions.
    polys : list, optional
        Combinatorial face definitions.

    Note
    ----
    The `verts`, `lines`, and `polys` arguments are ignored when
    `data` is a :class:`vtkPolyData` instance.
    """

    def __init__(self, data, *, verts=None, lines=None, polys=None):
        if isinstance(data, vtk.vtkPolyData):
            self._points = vtk_to_numpy(data.GetPoints().GetData())
            self._vtk_polydata = data
        else:
            self._points = np.asarray(data)
            self._vtk_polydata = vtk.vtkPolyData()

            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(self._points))

            self._vtk_polydata.SetPoints(points)

            if verts is not None:
                polydata = self._vtk_polydata
                cells = vtk.vtkCellArray()

                for v in verts:
                    cells.InsertNextCell(1, [int(v)])

                polydata.SetVerts(cells)

            if lines is not None:
                polydata = self._vtk_polydata
                cells = vtk.vtkCellArray()

                for e in lines:
                    edge = vtk.vtkIdList()

                    for v in e:
                        edge.InsertNextId(int(v))

                    cells.InsertNextCell(edge)

                polydata.SetLines(cells)

            if polys is not None:
                polydata = self._vtk_polydata
                cells = vtk.vtkCellArray()

                for f in polys:
                    face = vtk.vtkIdList()

                    for v in f:
                        face.InsertNextId(int(v))

                    cells.InsertNextCell(face)

                polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._vtk_polydata)

        # Create standard lookup table and how this table is used by
        # the mapper.
        mapper.SetLookupTable(_generic_lut(gradient='spectral'))
        mapper.SetUseLookupTableScalarRange(True)
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Initialize Prop base class. Mixins don't have an initializer.
        super().__init__(actor)

        # Initialize private properties.
        # self._draggable = None
        # self._scalars = None

    @property
    def points(self):
        """ Point coordinate array access.

        The contents of this array may be changed in place. Calling
        :meth:`modified` will update the visual representation on the
        next render pass.

        :type: ~numpy.ndarray
        """
        return self._points

    @property
    def color(self):
        """ Color property.

        Setting the global color attribute disables coloring using previously
        set scalars with :meth:`colorize`.

        :type: array_like, shape (3, )
        """
        return self._vtk_prop.GetProperty().GetColor()

    @color.setter
    def color(self, value):
        self._vtk_prop.GetMapper().SetScalarVisibility(False)
        self._vtk_prop.GetProperty().SetColor(value)

    def colorize(self, scalars, items, interpolate_scalars=False):
        r""" Colorize polygonal data.

        Colorize by assinging vertex colors or face colors. Vertex colors
        are interpolated across faces. Colors can be specified directly as
        RGB intensity triples or via a color map that maps scalar values
        to colors (see :meth:`lookuptable` for details on the mapping).

        Parameters
        ----------
        scalars : array_like
            Scalar values.
        items : str
            Either 'verts' or 'cells'.


        Note
        ----
        Use the :func:`colorbar` function to display a visual representation
        of the lookup table used for color mapping.


        .. rubric:: Examples

        Apply colors to the vertices of the unit cube :math:`[0, 1]^3`.
        Since vertex coordinates range between 0 and 1 we use them directly
        as RGB color specification.

        .. code-block:: python
           :linenos:

           import m3sh.vis as vis

           box = vis.box([0., 0., 0.], [1., 1., 1.])
           box.edges()
           box.colorize(box.points, 'verts')

           vis.show()


        Color the cube by :math:`z`-coordinate value. Not the difference
        when interpolating scalars before mapping them to colors.

        .. code-block:: python
           :linenos:

           import m3sh.vis as vis

           box = vis.box([0., 0., 0.], [1., 1., 1.])
           box.edges()
           box.colorize(box.points[:, 2], 'verts', interpolate_scalars=True)

           vis.colorbar(box)
           vis.show()


        To map colors to faces we have to specify a scalar for each of the
        six faces of the cube. Note that the `interpolate_scalars` argument
        has no effect when mapping colors to faces.

        .. code-block:: python
           :linenos:

           import m3sh.vis as vis

           box = vis.box([0., 0., 0.], [1., 1., 1.])
           box.edges()
           box.colorize([.0, .2, .4, .6, .8, 1.], 'cells')

           vis.colorbar(box)
           vis.show()
        """
        if items == 'verts':
            self._set_point_scalars(scalars)
        elif items == 'cells':
            self._set_cell_scalars(scalars)
        else:
            raise ValueError(f"invalid item argument '{items}'")

        self._vtk_prop.GetMapper().SetInterpolateScalarsBeforeMapping(
            interpolate_scalars)

    # def contour(self, scalars=None, *, levels=None, range=(None, None),
    #             width=None, style=None, color=None):
    #     """
    #     """
    #     if scalars is None and self._scalars is None:
    #         raise ValueError("required argument 'scalars' is missing")

    #     if scalars is not None:
    #         self._set_point_scalars(scalars)

    #     if not hasattr(self, '_contour'):
    #         self._contour = None

    #     if style == '':
    #         delete(self._contour)
    #         self._contour = None
    #         return

    #     if self._contour is None:
    #         lo = self._scalars.min() if range[0] is None else range[0]
    #         hi = self._scalars.max() if range[1] is None else range[1]

    #         curves = vtk.vtkContourFilter()
    #         curves.SetInputData(self._vtk_polydata)
    #         curves.GenerateValues(levels, lo, hi)

    #         mapper = vtk.vtkPolyDataMapper()
    #         mapper.SetInputConnection(curves.GetOutputPort())

    #         actor = vtk.vtkActor()
    #         actor.SetMapper(mapper)
    #         actor.SetPickable(False)

    #         add(actor)
    #         self._contour = actor
    #     else:
    #         actor = self._contour

    #     if width is not None:
    #         actor.GetProperty().SetLineWidth(width)

    #     if style == 'lines':
    #         actor.GetProperty().SetRenderLinesAsTubes(False)
    #     elif style == 'tubes':
    #         actor.GetProperty().SetRenderLinesAsTubes(True)

    #     if color == 'scalars':
    #         mapper.SetColorModeToMapScalars()
    #         mapper.SetLookupTable(self.mapper.GetLookupTable())
    #         mapper.SetUseLookupTableScalarRange(True)
    #         mapper.SetScalarVisibility(True)
    #     elif color is not None:
    #         mapper.SetScalarVisibility(False)
    #         actor.GetProperty().SetColor(color)

    # def verts(self, style=None, size=None, color=None):
    #     """ Vertex display.

    #     Set visual properties of vertices.

    #     Parameters
    #     ----------
    #     style : str, optional
    #         Either 'points' or 'spheres', :obj:`False` to disable.
    #     size : int, optional
    #         Size in pixels.
    #     color : array_like, shape (3, ), optional
    #         Vertex color.

    #     Note
    #     ----
    #     Parameters with a :obj:`None` value do not affect the corresponding
    #     vertex display property.

    #     Important
    #     ---------
    #     On some rendering backends vertex display only works when edges
    #     are displayed. For now use :func:`scatter` as a work-around.
    #     """
    #     if style == 'points':
    #         self._vtk_prop.GetProperty().SetRenderPointsAsSpheres(False)
    #         self._vtk_prop.GetProperty().SetVertexVisibility(True)
    #     elif style == 'spheres':
    #         self._vtk_prop.GetProperty().SetRenderPointsAsSpheres(True)
    #         self._vtk_prop.GetProperty().SetVertexVisibility(True)
    #     elif style == '' or style is False:
    #         self._vtk_prop.GetProperty().SetVertexVisibility(False)
    #     else:
    #         self._vtk_prop.GetProperty().SetVertexVisibility(True)

    #     if size is not None:
    #         self._vtk_prop.GetProperty().SetPointSize(size)

    #     if color is not None:
    #         self._vtk_prop.GetProperty().SetVertexColor(color)

    # def edges(self, style=None, width=None, color=None):
    #     """ Edge display.

    #     Set visual properties of edges.

    #     Parameters
    #     ----------
    #     style : str, optional
    #         Either 'lines' or 'tubes', :obj:`False` to disable.
    #     width : int, optional
    #         Edge width in pixels.
    #     color : array_like, shape (3, ), optional
    #         Edge color.

    #     Note
    #     ----
    #     Parameters with a :obj:`None` value do not affect the corresponding
    #     edge display property.
    #     """
    #     if style == 'lines':
    #         self._vtk_prop.GetProperty().SetRenderLinesAsTubes(False)
    #         self._vtk_prop.GetProperty().SetEdgeVisibility(True)
    #     elif style == 'tubes':
    #         self._vtk_prop.GetProperty().SetRenderLinesAsTubes(True)
    #         self._vtk_prop.GetProperty().SetEdgeVisibility(True)
    #     elif style == '' or style is False:
    #         self._vtk_prop.GetProperty().SetEdgeVisibility(False)
    #     else:
    #         self._vtk_prop.GetProperty().SetEdgeVisibility(True)

    #     if width is not None:
    #         self._vtk_prop.GetProperty().SetLineWidth(width)

    #     if color is not None:
    #         self._vtk_prop.GetProperty().SetEdgeColor(color)

    def modified(self):
        """ Update notification.

        Should be called when the contents of a shared data buffer are
        modified.

        Note
        ----
        Changing the **shape** of a shared data buffer is likely to
        result in a segmentation fault or other undefined behavior.
        """
        self._vtk_polydata.GetPoints().Modified()
        self._vtk_polydata.GetPointData().Modified()
        self._vtk_polydata.GetCellData().Modified()

    def _set_point_scalars(self, value):
        polydata = self._vtk_polydata
        polydata.GetCellData().SetScalars(None)

        self._set_scalars(polydata.GetPointData(),
                          polydata.GetNumberOfPoints(), value)

        self._vtk_prop.GetMapper().SetScalarModeToUsePointData()

    def _set_cell_scalars(self, value):
        polydata = self._vtk_polydata
        polydata.GetPointData().SetScalars(None)

        self._set_scalars(polydata.GetCellData(),
                          polydata.GetNumberOfCells(), value)

        self._vtk_prop.GetMapper().SetScalarModeToUseCellData()

    def _set_scalars(self, data, size, value):
        mapper = self._vtk_prop.GetMapper()

        if np.shape(value) == (size, ):
            self._scalars = np.asarray(value)

            data.SetScalars(numpy_to_vtk(self._scalars))
            mapper.SetColorModeToMapScalars()
            mapper.SetScalarVisibility(True)
        elif np.shape(value) == (size, 3):
            self._scalars = np.asarray(value)

            data.SetScalars(numpy_to_vtk(self._scalars))
            mapper.SetColorModeToDirectScalars()
            mapper.SetScalarVisibility(True)
        else:
            raise ValueError('wrong size of scalar array')

    # def _reset_scalars(self):
    #     polydata = self._prop.GetMapper().GetInput()

    #     data = polydata.GetPointData()
    #     data.SetScalars(None)
    #     data.Modified()

    #     data = polydata.GetCellData()
    #     data.SetScalars(None)
    #     data.Modified()

    #     self._prop.GetMapper().SetScalarVisibility(False)
    #     self._scalars = None


class __PolyData():
    """ Polygonal shape wrapper.

    Manages visual properties of a polygonal shape. Point clouds are
    polygonal shapes that only define 0-dimensional cells. If `data`
    is an `array_like` representation of a point set, it is converted
    to an equivalent :class:`~numpy.ndarray` instance. The created
    array is then accessible via the :attr:`points` attribute.

    Parameters
    ----------
    data : array_like or vtkPolyData
        Point data.
    verts : list, optional
        Combinatorial vertex definitions.
    lines : list, optional
        Not supported yet.
    faces : list, optional
        Combinatorial face definitions.

    Note
    ----
    The `verts`, `lines`, and `faces` arguments are ignored when
    `data` is a :class:`vtkPolyData` instance.
    """

    def __init__(self, data, *, verts=None, lines=None, faces=None):
        if isinstance(data, vtk.vtkPolyData):
            self._points = vtk_to_numpy(data.GetPoints().GetData())
            self._polydata = data
        else:
            self._points = np.asarray(data)
            self._polydata = vtk.vtkPolyData()

            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(self._points))

            polydata = self._polydata
            polydata.SetPoints(points)

            if verts is not None:
                cells = vtk.vtkCellArray()
                for v in verts:
                    cells.InsertNextCell(1, [int(v)])
                polydata.SetVerts(cells)

            # if lines is not None:
            #     cells = vtk.vtkCellArray()

            #     for i in range(len(edges) - 1):
            #         if lines[i] > -1 and lines[i+1] > -1:
            #             line = vtk.vtkIdList()
            #             line.InsertNextId(lines[i])
            #             line.InsertNextId(lines[i+1])
            #             cells.InsertNextCell(line)

            #     polydata.SetLines(cells)

            if faces is not None:
                cells = vtk.vtkCellArray()
                for f in faces:
                    face = vtk.vtkIdList()
                    for v in f:
                        face.InsertNextId(int(v))
                    cells.InsertNextCell(face)
                polydata.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._polydata)

        # Create standard lookup table and how this table is used
        # by the mapper.
        mapper.SetLookupTable(_generic_lut())
        mapper.SetUseLookupTableScalarRange(True)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Initialize base class.
        super().__init__(actor)

        # Initialize private properties.
        self._draggable = None
        self._scalars = None

    @property
    def points(self):
        """ Point coordinate array access.

        The contents of this array may be changed in place. Calling
        :meth:`modified` will update the visual representation on the
        next render pass.

        :type: ~numpy.ndarray
        """
        return self._points

    @property
    def polydata(self):
        """ Data access.

        Access VTK's polygonal shape definition.

        :type: vtkPolyData
        """
        return self._polydata
        # return self._prop.GetMapper().GetInput()

    @property
    def color(self):
        """ Color property.

        Setting the global color attribute disables coloring using previously
        set scalars with :meth:`colorize`.

        :type: array_like, shape (3, )
        """
        return self._prop.GetProperty().GetColor()

    @color.setter
    def color(self, value):
        self._prop.GetMapper().SetScalarVisibility(False)
        self._prop.GetProperty().SetColor(value)

    @property
    def scalars(self):
        """ Scalar access.

        Access scalar data set by the :meth:`colorize` method.

        :type: ~numpy.ndarray
        """
        self.modified()
        return self._scalars

    # @property
    # def draggable(self):
    #     return self._draggable

    # @draggable.setter
    # def draggable(self, value):
    #     self._draggable = value

    def colorize(self, scalars, items, range=None, gradient=None,
                 logscale=None, size=None):
        """ Colorize polygonal data.

        Colorize by assinging vertex colors or face colors. Vertex colors
        are interpolated across faces. Colors can be specified directly as
        RGB intensity triples or via a color map that maps scalar values
        to RGB values.

        Parameters
        ----------
        scalars : ~numpy.ndarray
            Scalar values. Either one scalar per item or one RGB color
            triple per item.
        items : str
            Either 'points' or 'cells'.
        range : (float, float), optional
            Lookup table range. Defaults to the range given by the
            smallest and largest scalar value.
        gradient : str, optional
            Color scheme identifier.
        logscale : bool, optional
            Toggle logarithmic scaling.
        size : int, optional
            Size of lookup table.


        Mapping scalars to colors uses a lookup table managed by the
        :attr:`mapper` instance of an actor. The `range`, `gradient`, and
        `logscale` arguments directly influence the lookup table. Lookup
        tables can be further customized via the :meth:`lookuptable` method.

        Use the :func:`colorbar` function to display a visual representation
        of a lookup table.

        Note
        ----
        When specifying colors directly by RGB triples, all arguments
        except `scalars` and `items` are ignored.
        """
        if scalars is not None:
            if items == 'points':
                self._set_point_scalars(scalars)
            elif items == 'cells':
                self._set_cell_scalars(scalars)
            else:
                raise ValueError(f"invalid item argument '{items}'")

            self.lookuptable(range, gradient, logscale, size)
        # else:
        #     self._reset_scalars()

    def verts(self, style=None, size=None, color=None):
        """ Vertex display.

        Set visual properties of vertices.

        Parameters
        ----------
        style : str, optional
            Either 'points' or 'spheres'. :obj:`False` to disable.
        size : int, optional
            Size in pixels.
        color : array_like, shape (3, ), optional
            Vertex color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        vertex display property.

        Important
        ---------
        On some rendering backends vertex display only works when edges
        are displayed. This might get fixed in future VTK releases. For
        now use :func:`scatter` as a work-around.
        """
        if style == 'points':
            self._prop.GetProperty().SetRenderPointsAsSpheres(False)
            self._prop.GetProperty().SetVertexVisibility(True)
        elif style == 'spheres':
            self._prop.GetProperty().SetRenderPointsAsSpheres(True)
            self._prop.GetProperty().SetVertexVisibility(True)
        elif style == '' or style is False:
            self._prop.GetProperty().SetVertexVisibility(False)
        else:
            self._prop.GetProperty().SetVertexVisibility(True)

        if size is not None:
            self._prop.GetProperty().SetPointSize(size)

        if color is not None:
            self._prop.GetProperty().SetVertexColor(color)

    def edges(self, style=None, width=None, color=None):
        """ Edge display.

        Set visual properties of edges.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes'. :obj:`False` to disable.
        width : int, optional
            Edge width in pixels.
        color : array_like, shape (3, ), optional
            Edge color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        edge display property.
        """
        if style == 'lines':
            self._prop.GetProperty().SetRenderLinesAsTubes(False)
            self._prop.GetProperty().SetEdgeVisibility(True)
        elif style == 'tubes':
            self._prop.GetProperty().SetRenderLinesAsTubes(True)
            self._prop.GetProperty().SetEdgeVisibility(True)
        elif style == '' or style is False:
            self._prop.GetProperty().SetEdgeVisibility(False)
        else:
            self._prop.GetProperty().SetEdgeVisibility(True)

        if width is not None:
            self._prop.GetProperty().SetLineWidth(width)

        if color is not None:
            self._prop.GetProperty().SetEdgeColor(color)

    def silhouette(self, style=None, width=None, color=None):
        """ Silhouette display.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes'. :obj:`False` to disable.
        width : int, optional
            Edge width in pixels.
        color : array_like, shape (3, ), optional
            Edge color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        silhouette display property.
        """
        if not hasattr(self, '_silhouette'):
            self._silhouette = None

        if style == '' or style is False:
            delete(self._silhouette)
            self._silhouette = None
            return

        if self._silhouette is None:
            outline = vtk.vtkPolyDataSilhouette()
            outline.SetInputData(self.polydata)
            outline.SetCamera(_renderer.GetActiveCamera())
            outline.SetEnableFeatureAngle(False)
            outline.SetBorderEdges(True)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPickable(False)

            if color is None:
                actor.GetProperty().SetColor(colors.black)

            if width is None:
                actor.GetProperty().SetLineWidth(2)

            add(actor)
            self._silhouette = actor
        else:
            actor = self._silhouette

        if width is not None:
            actor.GetProperty().SetLineWidth(width)

        if style == 'lines':
            actor.GetProperty().SetRenderLinesAsTubes(False)
        elif style == 'tubes':
            actor.GetProperty().SetRenderLinesAsTubes(True)

        if color is not None:
            actor.GetProperty().SetColor(color)

    def modified(self):
        """ Update notification.

        Should be called when the contents of a shared data buffer are
        modified.

        Note
        ----
        Changing the **shape** of a shared data buffer is likely to
        result in a segmentation fault or other undefined behavior.
        """
        self._prop.GetMapper().GetInput().GetPoints().Modified()
        self._prop.GetMapper().GetInput().GetPointData().Modified()

        # There are cells that define vertices and lines. None of the
        # wrapped shapes use them.
        self._prop.GetMapper().GetInput().GetPolys().Modified()
        self._prop.GetMapper().GetInput().GetCellData().Modified()

    def _set_point_scalars(self, value):
        polydata = self._prop.GetMapper().GetInput()

        self._set_scalars(polydata.GetPointData(),
                          polydata.GetNumberOfPoints(), value)

        polydata.GetCellData().SetScalars(None)
        polydata.GetCellData().Modified()

        self._prop.GetMapper().SetScalarModeToUsePointData()

    def _set_cell_scalars(self, value):
        polydata = self._prop.GetMapper().GetInput()

        self._set_scalars(polydata.GetCellData(),
                          polydata.GetNumberOfCells(), value)

        polydata.GetPointData().SetScalars(None)
        polydata.GetPointData().Modified()

        self._prop.GetMapper().SetScalarModeToUseCellData()

    def _set_scalars(self, data, size, value):
        """ Set scalar array.

        Parameters
        ----------
        data :
        size :
        value : ~numpy.ndarray
        """
        mapper = self._prop.GetMapper()

        if np.shape(value) == (size, ):
            self._scalars = np.asarray(value)

            data.SetScalars(numpy_to_vtk(self._scalars))
            data.Modified()

            mapper.SetColorModeToMapScalars()
            mapper.SetScalarVisibility(True)
        elif np.shape(value) == (size, 3):
            self._scalars = np.asarray(value)

            data.SetScalars(numpy_to_vtk(self._scalars))
            data.Modified()

            mapper.SetColorModeToDirectScalars()
            mapper.SetScalarVisibility(True)
        else:
            raise ValueError('wrong size of scalar array')

    def _reset_scalars(self):
        polydata = self._prop.GetMapper().GetInput()

        data = polydata.GetPointData()
        data.SetScalars(None)
        data.Modified()

        data = polydata.GetCellData()
        data.SetScalars(None)
        data.Modified()

        self._prop.GetMapper().SetScalarVisibility(False)
        self._scalars = None


class _PolyMesh(PolyData):
    """ Polygonal mesh shape.

    Wrapper class managing the visual properties of a mesh. Instances of
    this class are typically generated using the :func:`mesh` function.

    Parameters
    ----------
    mesh : Mesh or tuple
        Polygonal mesh representation.

    Raises
    ------
    ValueError
        When `copy` evaluates to :obj:`False` and a copy cannot be
        avoided, e.g., when vertex coordinates are represented as a
        nested list of floating point values.

    Note
    ----
    The generated :class:`PolyMesh` instance and `mesh` share their
    vertex coordinate data buffers if not disabled explicitly.
    """

    def __init__(self, mesh):
        super().__init__(mesh[0], faces=mesh[1])

        # Initialize private properties.
        self._mesh = mesh

    @property
    def mesh(self):
        """ Mesh access.

        :type: Mesh
        """
        return self._mesh

    def texture(self, img, uv):
        """ Texture mapping.

        Parameters
        ----------
        img : str
            Name of image file.
        uv : ~numpy.ndarray
            Texture coordinates.
        """
        self.polydata.GetPointData().SetTCoords(numpy_to_vtk(uv))

        factory = vtk.vtkImageReader2Factory()
        reader = factory.CreateImageReader2(img)
        reader.SetFileName(img)
        reader.Update()

        texture = vtk.vtkTexture()
        texture.InterpolateOn()
        texture.SetInputConnection(reader.GetOutputPort())

        self.prop.SetTexture(texture)

    def _contour(self, scalars=None, *, levels=None, range=(None, None),
                width=None, style=None, color=None):
        """
        """
        if scalars is None and self._scalars is None:
            raise ValueError("required argument 'scalars' is missing")

        if scalars is not None:
            self._set_point_scalars(scalars)

        if not hasattr(self, '_contour'):
            self._contour = None

        if style == '':
            delete(self._contour)
            self._contour = None
            return

        if self._contour is None:
            lo = self._scalars.min() if range[0] is None else range[0]
            hi = self._scalars.max() if range[1] is None else range[1]

            curves = vtk.vtkContourFilter()
            curves.SetInputData(self.polydata)
            curves.GenerateValues(levels, lo, hi)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(curves.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPickable(False)

            add(actor)
            self._contour = actor
        else:
            actor = self._contour

        if width is not None:
            actor.GetProperty().SetLineWidth(width)

        if style == 'lines':
            actor.GetProperty().SetRenderLinesAsTubes(False)
        elif style == 'tubes':
            actor.GetProperty().SetRenderLinesAsTubes(True)

        if color == 'scalars':
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(self.mapper.GetLookupTable())
            mapper.SetUseLookupTableScalarRange(True)
            mapper.SetScalarVisibility(True)
        elif color is not None:
            mapper.SetScalarVisibility(False)
            actor.GetProperty().SetColor(color)


class PolyMesh(PolyData):
    """ Polygonal mesh shape.

    Wrapper class managing the visual properties of a mesh. Instances of
    this class are typically generated using the :func:`mesh` function.

    Parameters
    ----------
    mesh : Mesh or tuple
        Polygonal mesh representation.

    Note
    ----
    The generated :class:`PolyMesh` instance and `mesh` share their
    vertex coordinate data buffers.
    """

    def __init__(self, mesh):
        if isinstance(mesh, vtk.vtkPolyData):
            super().__init__(mesh)

            # There is not halfedge mesh instance when initializing from
            # a vtkPolyData instance.
            self._mesh = None
            self._normals = None

            if (normals := mesh.GetPointData().GetNormals()) is not None:
                self._normals = vtk_to_numpy(normals)
        else:
            # We rely on the __getitem__ method of the Mesh class to get
            # the array of point coordintaes and an iterable that can be
            # treated as list[list[int]] to specify mesh connectivity.
            super().__init__(mesh[0], polys=mesh[1])

            # Initialize private instance attributes. Can be accessed as
            # properties of the same name.
            self._mesh = mesh
            self._normals = None

    @property
    def mesh(self):
        """ Mesh access.

        :type: Mesh
        """
        return self._mesh

    @property
    def normals(self):
        """ Vertex normals.

        Vertex normals are merely a visualization hint and result in
        smooth shading of the mesh.

        :type: ~numpy.ndarray
        """
        return self._normals

    @normals.setter
    def normals(self, value):
        data = self._vtk_polydata.GetPointData()

        if value is not None:
            self._normals = np.asarray(value)
            value = numpy_to_vtk(self._normals)
        else:
            self._normals = None

        data.SetNormals(value)

    def verts(self, style=None, size=None, color=None):
        """ Vertex display.

        Set visual properties of vertices.

        Parameters
        ----------
        style : str, optional
            Either 'points' or 'spheres', :obj:`False` to disable.
        size : int, optional
            Size in pixels.
        color : array_like, shape (3, ), optional
            Vertex color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        vertex display property.

        Important
        ---------
        On some rendering backends vertex display only works when edges
        are displayed. For now use :func:`scatter` as a work-around.
        """
        if style == 'points':
            self._vtk_prop.GetProperty().SetRenderPointsAsSpheres(False)
            self._vtk_prop.GetProperty().SetVertexVisibility(True)
        elif style == 'spheres':
            self._vtk_prop.GetProperty().SetRenderPointsAsSpheres(True)
            self._vtk_prop.GetProperty().SetVertexVisibility(True)
        elif style == '' or style is False:
            self._vtk_prop.GetProperty().SetVertexVisibility(False)
        else:
            self._vtk_prop.GetProperty().SetVertexVisibility(True)

        if size is not None:
            self._vtk_prop.GetProperty().SetPointSize(size)

        if color is not None:
            self._vtk_prop.GetProperty().SetVertexColor(color)

    def edges(self, style=None, width=None, color=None):
        """ Edge display.

        Set visual properties of edges.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes', :obj:`False` to disable.
        width : int, optional
            Edge width in pixels.
        color : array_like, shape (3, ), optional
            Edge color.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        edge display property.
        """
        if style == 'lines':
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(False)
            self._vtk_prop.GetProperty().SetEdgeVisibility(True)
        elif style == 'tubes':
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(True)
            self._vtk_prop.GetProperty().SetEdgeVisibility(True)
        elif style == '' or style is False:
            self._vtk_prop.GetProperty().SetEdgeVisibility(False)
        else:
            self._vtk_prop.GetProperty().SetEdgeVisibility(True)

        if width is not None:
            self._vtk_prop.GetProperty().SetLineWidth(width)

        if color is not None:
            self._vtk_prop.GetProperty().SetEdgeColor(color)

    # def texture(self, img, uv):
    #     """ Texture mapping.

    #     Parameters
    #     ----------
    #     img : str
    #         Name of image file.
    #     uv : ~numpy.ndarray
    #         Texture coordinates.
    #     """
    #     self.polydata.GetPointData().SetTCoords(numpy_to_vtk(uv))

    #     factory = vtk.vtkImageReader2Factory()
    #     reader = factory.CreateImageReader2(img)
    #     reader.SetFileName(img)
    #     reader.Update()

    #     texture = vtk.vtkTexture()
    #     texture.InterpolateOn()
    #     texture.SetInputConnection(reader.GetOutputPort())

    #     self.prop.SetTexture(texture)

    # def contour(self, scalars=None, *, levels=None, range=(None, None),
    #             width=None, style=None, color=None):
    #     """
    #     """
    #     if scalars is None and self._scalars is None:
    #         raise ValueError("required argument 'scalars' is missing")

    #     if scalars is not None:
    #         self._set_point_scalars(scalars)

    #     if not hasattr(self, '_contour'):
    #         self._contour = None

    #     if style == '':
    #         delete(self._contour)
    #         self._contour = None
    #         return

    #     if self._contour is None:
    #         lo = self._scalars.min() if range[0] is None else range[0]
    #         hi = self._scalars.max() if range[1] is None else range[1]

    #         curves = vtk.vtkContourFilter()
    #         curves.SetInputData(self._vtk_polydata)
    #         curves.GenerateValues(levels, lo, hi)

    #         mapper = vtk.vtkPolyDataMapper()
    #         mapper.SetInputConnection(curves.GetOutputPort())

    #         actor = vtk.vtkActor()
    #         actor.SetMapper(mapper)
    #         actor.SetPickable(False)

    #         add(actor)
    #         self._contour = actor
    #     else:
    #         actor = self._contour

    #     if width is not None:
    #         actor.GetProperty().SetLineWidth(width)

    #     if style == 'lines':
    #         actor.GetProperty().SetRenderLinesAsTubes(False)
    #     elif style == 'tubes':
    #         actor.GetProperty().SetRenderLinesAsTubes(True)

    #     if color == 'scalars':
    #         mapper.SetColorModeToMapScalars()
    #         mapper.SetLookupTable(self.mapper.GetLookupTable())
    #         mapper.SetUseLookupTableScalarRange(True)
    #         mapper.SetScalarVisibility(True)
    #     elif color is not None:
    #         mapper.SetScalarVisibility(False)
    #         actor.GetProperty().SetColor(color)

    def _avg_edge_length(self):
        # The edge extraction filter only works when there are polygonal
        # cells defined! This cannot be used to compute the average edge
        # length of a PolyGraph instance!
        filter = vtk.vtkExtractEdges()
        filter.SetInputData(self._vtk_polydata)
        filter.Update()

        points = vtk_to_numpy(filter.GetOutput().GetPoints().GetData())
        edges = filter.GetOutput().GetLines()
        edge_iter = edges.NewIterator()

        min, max, sum = np.inf, 0.0, 0.0

        while not edge_iter.IsDoneWithTraversal():
            edge = edge_iter.GetCurrentCell()
            i, j = edge.GetId(0), edge.GetId(1)

            length = np.linalg.norm(points[i, :] - points[j, :])
            sum += length

            min = length if length < min else min
            max = length if length > max else max

            edge_iter.GoToNextCell()

        return sum / edges.GetNumberOfCells(), min, max


class PolyGraph(PolyData):
    """ Polygonal graph.

    Parameters
    ----------
    nodes : array_like
        Node coordinate array.
    graph : csr_matrix
        Combinatorial graph definition.
    """

    def __init__(self, nodes, graph):
        # Raises an attribute error if not a sparse matrix or array type.
        assert graph.format == 'csr'

        # Assume csr_matrix representation of graph, i.e., edge defined by
        # the sparsity pattern, not by edge weights!
        rowptr = graph.indptr
        colidx = graph.indices

        # For a symmetric matrix this will define all edges twice! Fix!
        edges = ((i, j) for i in range(len(rowptr)-1)
                        for j in colidx[rowptr[i]:rowptr[i+1]])

        super().__init__(nodes, lines=edges)

        # mapper = self._vtk_prop.GetMapper()
        # mapper.SetResolveCoincidentTopologyToPolygonOffset()

    def edges(self, style=None, width=None):
        """ Edge display.

        Set visual properties of edges.

        Parameters
        ----------
        style : str, optional
            Either 'lines' or 'tubes'.
        width : int, optional
            Edge width in pixels.

        Note
        ----
        Parameters with a :obj:`None` value do not affect the corresponding
        edge display property.
        """
        if style == 'lines':
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(False)
        elif style == 'tubes':
            self._vtk_prop.GetProperty().SetRenderLinesAsTubes(True)

        if width is not None:
            self._vtk_prop.GetProperty().SetLineWidth(width)


class LookupTable(Prop):
    """ Color bar.

    Visual representation of a lookup table associated with a displayed
    shape.

    Parameters
    ----------
    actor : Prop or vtkActor
        A render object.

    Note
    ----
    The wrapped actor is a 2-dimensional actor.
    """

    def __init__(self, actor):
        if isinstance(actor, Prop):
            actor = actor.prop

        lut = actor.GetMapper().GetLookupTable()

        actor = vtk.vtkScalarBarActor()
        actor.SetDrawBelowRangeSwatch(lut.GetUseBelowRangeColor())
        actor.SetDrawAboveRangeSwatch(lut.GetUseAboveRangeColor())

        actor.SetNumberOfLabels(5)
        actor.SetBarRatio(0.2)
        actor.SetMaximumWidthInPixels(180)
        actor.GetLabelTextProperty().SetFontSize(14)
        actor.SetUnconstrainedFontSize(True)
        actor.SetLookupTable(lut)

        super().__init__(actor)

    @property
    def position(self):
        """ Position property.

        Location of the lower left corner in normalized screen
        coordinates.
        """
        return self.prop.GetPosition()

    @position.setter
    def position(self, value):
        self.prop.SetPosition(value)
        # self.prop.Modified()

    # def _modified(self, object=None):
    #     """ Update representation.

    #     Update lookup table representation from `object`.

    #     Parameters
    #     ----------
    #     object : Shape, optional
    #         Polygonal shape instance.
    #     """
    #     if object is None:
    #         lut = self.prop.GetLookupTable()
    #     else:
    #         try:
    #             actor = object.actor
    #         except AttributeError:
    #             actor = object

    #         lut = actor.GetMapper().GetLookupTable()
    #         self.actor.SetLookupTable(lut)

    #     self.actor.SetDrawBelowRangeSwatch(lut.GetUseBelowRangeColor())
    #     self.actor.SetDrawAboveRangeSwatch(lut.GetUseAboveRangeColor())


# class RenderMesh(RenderObject):
#     """
#     """

#     def __init__(self, obj, cpts=None):
#         """
#         """
#         super().__init__(obj)

#         self._cpts = cpts
#         self._usecpts = False

#     @RenderObject.pickable.setter
#     def pickable(self, value):
#         """
#         """
#         if self._usecpts:
#             self._cpts.SetPickable(bool(value))
#         else:
#             self._actor.SetPickable(bool(value))

#     @property
#     def usecpts(self):
#         """
#         """
#         return self._usecpts

#     @usecpts.setter
#     def usecpts(self, value):
#         """
#         """
#         if self._cpts is not None:
#             self._usecpts = bool(value)

#             if self._usecpts:
#                 self._cpts.SetPickable(self._actor.GetPickable())
#                 self._actor.SetPickable(False)
#             else:
#                 self._actor.SetPickable(self._cpts.GetPickable())
#                 self._cpts.SetPickable(False)


class _MouseInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """ Trackball camera interactor style.

    Initialize state variables and register user defined callbacks for
    mouse and keyboard events.

    Parameters
    ----------
    lmbdown : list[callable]
        Left button press callbacks.
    lmbup : list[callable]
        Left button release callbacks.
    rmbdown : list[callable]
        Right button press callbacks.
    rmbup : list[callable]
        Right button release callbacks.
    keydown : list[callable]
        Key press callbacks.
    keyup : list[callable]
        Key release callbacks.
    mousemove : list[callable]
        Mouse move callbacks.

    Note
    ----
    Multiple callbacks for the same type of event are invoked in the
    specified order. See :py:func:`show` for an example on how to define
    and use callback functions.
    """
    def __init__(self, lmbdown=None, lmbup=None,
                       rmbdown=None, rmbup=None,
                       keydown=None, keyup=None, mousemove=None):
        # User defined callbacks. Those callbacks are invoked in the order
        # they were passed.
        self._lmb_down_cbs = lmbdown if lmbdown is not None else []
        self._rmb_down_cbs = rmbdown if rmbdown is not None else []
        self._key_down_cbs = keydown if keydown is not None else []

        self._lmb_up_cbs = lmbup if lmbup is not None else []
        self._rmb_up_cbs = rmbup if rmbup is not None else []
        self._key_up_cbs = keyup if keyup is not None else []

        self._mouse_move_cbs = mousemove if mousemove is not None else []

        # See the vtkCommand class documentation for an enumeration of all
        # available events and their names.
        self.AddObserver("LeftButtonPressEvent", self._lmb_down_event)
        self.AddObserver("LeftButtonReleaseEvent", self._lmb_up_event)
        self.AddObserver("RightButtonPressEvent", self._rmb_down_event)
        self.AddObserver("RightButtonReleaseEvent", self._rmb_up_event)
        self.AddObserver("KeyPressEvent", self._key_press_event)
        self.AddObserver("KeyReleaseEvent", self._key_release_event)
        self.AddObserver("MouseMoveEvent", self._mouse_move_event)
        self.AddObserver("InteractionEvent", self._interaction_event)
        self.AddObserver("CharEvent", self._char_event)

    def _project(self, ren, x, y):
        """ Display to world coordinate projection.

        Helper function mainly used during vertex dragging. Relies on the
        value of :py:attr:`_picked_fac`.

        Parameters
        ----------
        ren : vtkRenderer
            The affected renderer or viewport.
        x : int
            Display x-coordinate.
        y : int
            Display y-coordinate.

        Returns
        -------
        ndarray
            World coordinates of the mapped point.

        Note
        ----
        Display coordinates are given in pixels and determine a point in the
        active render window.
        """
        # Convert from display to world coordinates. Pixel coordinates are
        # transformed to 3D world coordinates of the corresponding point on
        # the image plane.
        p2d = vtk.vtkCoordinate()
        p2d.SetCoordinateSystemToDisplay()
        p2d.SetValue(x, y)
        p2d = np.array(p2d.GetComputedWorldValue(ren))

        # Now apply the intercept theorem. Need the eye position for that.
        # If the scale factor is 1.0 the coordinates of the corresponding
        # point on the image plane are returned.
        eye = np.array(ren.GetActiveCamera().GetPosition())
        return eye + self._picked_fac*(p2d-eye)

    def _lmb_down_event(self, irenstyle, event):
        """ LMB down event handler.

        Dispatches left mouse button press events to user defined callbacks.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as ``self``.
        event : str
            String identifier of the event: ``'LeftButtonPressEvent'``
        """
        # Get corresponding render window interactor and event location
        # in window coordinates.
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        # Broadcast the event to all user defined callbacks.
        for cb in self._lmb_down_cbs:
            cb(iren, x, y, event=event)

        # Forward event to the standard event handler. Without this call
        # the scene cannot be rotated.
        self.OnLeftButtonDown()

    def _lmb_up_event(self, irenstyle, event):
        """ LMB up event handler.

        Dispatch left mouse button release events to user defined callbacks.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as ``self``.
        event : str
            String identifier of the event: ``'LeftButtonReleaseEvent'``
        """
        # Get corresponding render window interactor and event location
        # in window coordinates.
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        # Broadcast the event to all user defined callbacks.
        for cb in self._lmb_up_cbs:
            cb(iren, x, y, event=event)

        # Hand off to standard event handler. Without this call we are
        # stuck in scene rotation mode.
        self.OnLeftButtonUp()

    def _rmb_down_event(self, irenstyle, event):
        """ RMB down event handler.

        Dispatch to user defined event callbacks. On a successful point pick
        action the interactor style changes to vertex dragging. This stops
        once the right mouse button is released.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as ``self``.
        event : str
            String identifier of the event: ``'RightButtonPressEvent'``
        """
        # Get corresponding render window interactor, viewport and event
        # location in window coordinates.
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        # Broadcast the event to all user defined callbacks.
        for cb in self._rmb_down_cbs:
            cb(iren, x, y, event=event)

        # Forward events to the standard event handler. Without this call
        # zooming with the right mouse button is diabled.
        self.OnRightButtonDown()

    def _rmb_up_event(self, irenstyle, event):
        """ RMB up event handler.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as ``self``.
        event : str
            String identifier of the event: ``'RightButtonReleaseEvent'``
        """
        # Get corresponding render window interactor and event location
        # in window coordinates.
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        # Broadcast the event to all user defined callbacks.
        for cb in self._rmb_up_cbs:
            cb(iren, x, y, event=event)

        # Hand off to the standard event handler. Without this call we
        # are stuck in zoom mode.
        self.OnRightButtonUp()

    def _mouse_move_event(self, irenstyle, event):
        """ Mouse move event observer.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as `self`.
        event : str
            String identifier of the event: 'MouseMoveEvent'.
        """
        # Get corresponding render window interactor, viewport and event
        # location in window coordinates.
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        # Get the affected viewport to synchronize the axes widget to the
        # current camera.
        ren = iren.FindPokedRenderer(x, y)

        # Make the axes widget jump to the active viewport. While the
        # orientation marker widget placement is relative to the renderer,
        # the camera orientation widget can only be placed in one of the
        # corners of the render window.
        widget = iren.axes_widget

        try:
            parent = widget.GetParentRenderer()
            setren = widget.SetParentRenderer
        except AttributeError:
            parent = widget.GetCurrentRenderer()
            setren = widget.SetCurrentRenderer

        # Set parent (or current) renderer and request render pass for
        # the changes to take effect.
        if parent is not ren:
            setren(ren)
            iren.Render()

        # Broadcast the event to all user defined callbacks.
        for cb in self._mouse_move_cbs:
            cb(iren, x, y, event=event)

        # Hand off to the standard event handler.
        self.OnMouseMove()

    def _key_press_event(self, irenstyle, event):
        """ Key press event observer.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as `self`.
        event : str
            String identifier of the event: 'KeyPressEvent'.
        """
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        key = iren.GetKeySym().lower()

        if key == 'tab':
            iren.axes_widget.SetEnabled(not iren.axes_widget.GetEnabled())
            iren.Render()

        if key == 'h':
            _commands()

        for cb in self._key_down_cbs:
            cb(iren, x, y, event=event)

        self.OnKeyPress()

    def _key_release_event(self, irenstyle, event):
        """ Key release event observer.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as `self`.
        event : str
            String identifier of the event: 'KeyReleaseEvent'.
        """
        iren = irenstyle.GetInteractor()
        x, y = iren.GetEventPosition()

        for cb in self._key_up_cbs:
            cb(iren, x, y, event=event)

        self.OnKeyRelease()

    def _char_event(self, irenstyle, event):
        """ Character event observer.

        Any button press qualifies as a character event (including
        modifier keys). Character events trigger window rendering.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as `self`.
        event : str
            String identifier of the event: 'CharEvent'.

        Note
        ----
        This observer can be used to disable all default interactor
        keyboard commands.
        """
        # Keypresses typically invoke some computation. Start render pass
        # to update the scene.
        for iren in _interactors:
            iren.Render()

        # Uncomment this line to enable all default commands of the
        # interactor, like 'q' for quit and 'w' for wireframe rendering.
        self.OnChar()

    def _interaction_event(self, irenstyle, event):
        """ Interaction event observer.

        Dragging the mouse (moving the cursor while holding down a mouse
        button) qualifies as an interaction event.

        Parameters
        ----------
        irenstyle : vtkInteractorStyle
            The corresponding interactor style, same as `self`.
        event : str
            String identifier of the event: 'InteractionEvent'.

        Note
        ----
        Interaction events trigger a window rendering pass.
        """
        # Synchronize windows in case some of their renderers share a
        # camera (otherwise windows would first get updated once they
        # get back mouse focus). This is purely cosmetic!
        # for iren in _interactors:
        #     iren.Render()


class _VertexDragger:
    def initiate(self, iren, x, y):
        ren = iren.FindPokedRenderer(x, y)

        # If we are not already vertex dragging we start vertex dragging.
        # There cannot be a right button press event in this state.
        if not self._active:
            # A point picker will pick points of the underlying point set
            # of a data set even if those points not part of any cell.
            #picker = vtk.vtkPointPicker()
            picker = vtk.vtkCellPicker()
            # picker.SetTolerance(0.005)
            picker.Pick(x, y, 0, ren)
            obj = picker.GetActor()

            if obj is not None:
                # Check if we may drag the picked point. If the attribute is
                # not present all points can be dragged.
                pid = picker.GetPointId()
                print(pid)

                try:
                    draggable = pid in obj.draggable_points
                except AttributeError:
                    draggable = True
                except TypeError:
                    draggable = False

                if draggable:
                    # The picked point set is reachable by the picked actor's
                    # mapper object.
                    mapper = picker.GetMapper()

                    # Enter vertex dragging state. Store all information as
                    # attributes of the interactor style object.
                    self._vertex_drag = True
                    self._picked_obj = obj
                    self._picked_pts = mapper.GetInput().GetPoints()

                    p2d = vtk.vtkCoordinate()
                    p2d.SetCoordinateSystemToDisplay()
                    p2d.SetValue(x, y)
                    p2d = np.array(p2d.GetComputedWorldValue(ren))
                    p3d = np.array(picker.GetPickPosition())

                    # Eye point position and scale factor derived from the
                    # intercept thoerem.
                    eye = np.array(ren.GetActiveCamera().GetPosition())
                    fac = np.linalg.norm(p3d-eye)/np.linalg.norm(p2d-eye)

                    # World space coordinates of the point on the image
                    # plane and of the picked point.
                    self._picked_p2d = p2d
                    self._picked_p3d = p3d
                    self._picked_pid = pid
                    self._picked_fac = fac

                    # Broadcast the event to all user defined callbacks.
                    # Pass extra vertex dragging related keyword arguments.
                    for cb in self._rmb_down_cbs:
                        cb(iren, x, y, event=event, obj=obj,
                                                    pid=pid,
                                                    p3d=p3d)

                    # Do not execute standard event handlers. This would
                    # interfere with vertex dragging behavior.
                    return

            # Broadcast the event to all user defined callbacks. This path
            # is taken if nothing was picked.
            for cb in self._rmb_down_cbs:
                cb(iren, x, y, event=event)

            # Forward events to the standard event handlers when we are not
            # in vertex drag mode.
            self.OnRightButtonDown()
        else:
            # This point should never be reached! If it happens events were
            # received from the window system in asynchronous order.
            msg = ('Right mouse button press event received before ' +
                   'it was released!')
            raise RuntimeError(msg)

    def drag(self, iren, x, y):
        ren = iren.FindPokedRenderer(x, y)

        # Update the position of actively dragged points on mouse movement.
        # We are in this state as long as the right mouse button is down.
        if self._active:
            p3d = self._project(ren, x, y)
            pid = self._picked_pid
            obj = self._picked_obj

            # An actor can provide a project method. This method acts as a
            # constraint during point dragging.
            try:
                p3d = np.asarray(obj.project(ren, pid, p3d))
            except AttributeError:
                pass

            # Update point location of the point in the corresponding
            # point set. Calling .Modified() seems not enough to update the
            # render window.
            self._picked_pts.SetPoint(pid, p3d[0], p3d[1], p3d[2])
            self._picked_pts.Modified()

            # Nudge for render window update...
            ren.GetRenderWindow().Render()

            # Broadcast the event to all user defined callbacks. Pass extra
            # keyword argument during vertex dragging.
            for cb in self._mouse_move_cbs:
                cb(iren, x, y,
                   event='MouseDragEvent', obj=obj, pid=pid, p3d=p3d)

            # Invoke the modified event. Pass the id of the modified point
            # as additinal callback data. In order to receive that data the
            # callback needs to use a corresponding decorator, e.g.
            # @vtk.calldata_type(vtk.VTK_INT) to pass integers as data.
            # self._picked_obj.Modified()
            self._picked_obj.InvokeEvent('ModifiedEvent', pid)

            # Do not execute standard event handlers. They interfere with
            # vertex dragging behavior.
            return
        else:
            # Broadcast the event to all user defined callbacks. No extra
            # argument except event identifier.
            for cb in self._mouse_move_cbs:
                cb(iren, x, y, event=event)

    def finalize(self, iren, x, y):
        ren = iren.FindPokedRenderer(x, y)

        # This is the final update of the position of actively dragged points
        # on mouse movement.
        if self._vertex_drag:
            p3d = self._project(ren, x, y)
            pid = self._picked_pid
            obj = self._picked_obj

            # An actor can provide a project method. This method acts as a
            # constraint during point dragging.
            # try:
            #     p3d = np.asarray(obj.project(ren, pid, p3d))
            # except AttributeError:
            #     pass

            # Update point location of the point in the corresponding point
            # set. Call .Modified() to queue render window update.
            # self._picked_pts.SetPoint(pid, p3d[0], p3d[1], p3d[2])
            # self._picked_pts.Modified()

            # Calling .Modified is not always enough. Nudge...
            ren.GetRenderWindow().Render()

            # Broadcast the event to all user defined callbacks with extra
            # vertex dragging keyword arguments.
            for cb in self._rmb_up_cbs:
                cb(iren, x, y,
                   event='MouseDragEnd', obj=obj, pid=pid, p3d=p3d)

            # Invoke the modified event. Pass the id of the modified point
            # as additinal callback data. In order to receive that data the
            # callback needs to use a corresponding decorator, e.g.
            # @vtk.calldata_type(vtk.VTK_INT) to pass integers as data.
            # self._picked_obj.InvokeEvent('ModifiedEvent', pid)
        else:
            # Broadcast the event to all user defined callbacks. No extra
            # keyword arguments.
            for cb in self._rmb_up_cbs:
                cb(iren, x, y, event=event)

        # Leave vertex dragging mode and hand off to standard event handlers.
        # Reset the scale factor. The _project method will now map 2D window
        # coordinates to 3D image plane point coordinates.
        self._vertex_drag = False
        self._picked_obj = None
        self._picked_fac = 1.0


if __name__ == '__main__':
    _main()
    # _test()