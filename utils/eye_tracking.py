# let's load the eye gaze data first
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot, image
import matplotlib


# # # # #
# LOOK

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLS = {"butter": ['#fce94f',
                   '#edd400',
                   '#c4a000'],
        "orange": ['#fcaf3e',
                   '#f57900',
                   '#ce5c00'],
        "chocolate": ['#e9b96e',
                      '#c17d11',
                      '#8f5902'],
        "chameleon": ['#8ae234',
                      '#73d216',
                      '#4e9a06'],
        "skyblue": ['#729fcf',
                    '#3465a4',
                    '#204a87'],
        "plum": 	['#ad7fa8',
                  '#75507b',
                  '#5c3566'],
        "scarletred": ['#ef2929',
                       '#cc0000',
                       '#a40000'],
        "aluminium": ['#eeeeec',
                      '#d3d7cf',
                      '#babdb6',
                      '#888a85',
                      '#555753',
                      '#2e3436'],
        }
# FONT
# FONT = {	'family': 'Ubuntu',
# 		'size': 12}
# matplotlib.rc('font', **FONT)


def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                                    with a size of dispsize, and an image drawn onto it
                                    if an imagefile was passed
    """
    _, ext = os.path.splitext(imagefile)
    ext = ext.lower()
    data_type = 'float32' if ext == '.png' else 'uint8'
    screen = np.zeros((dispsize[1], dispsize[0]), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception(
                "ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if not os.name == 'nt':
            img = numpy.flipud(img)
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0]/2 - w/2)
        y = int(dispsize[1]/2 - h/2)

        # draw the image on the screen
        screen[y:y+h, x:x+w] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    # ax.axis([dispsize[0], 0, dispsize[1], 0])
    ax.imshow(screen, cmap="gray")  # , origin='upper')
    return fig, ax


def draw_fixations(fix, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5, savefilename=None):
    """Draws circles on the fixation locations, optionally on top of an image,
    with optional weigthing of the duration for circle size and colour

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                                    as produced by edfreader.read_edf, e.g.
                                    edfdata[trialnr]['events']['Efix']
    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)
    durationsize	-	Boolean indicating whether the fixation duration is
                                    to be taken into account as a weight for the circle
                                    size; longer duration = bigger (default = True)
    durationcolour	-	Boolean indicating whether the fixation duration is
                                    to be taken into account as a weight for the circle
                                    colour; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                                    the heatmap, where 0 is completely transparant and 1
                                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                                    fixations
    """

    img = image.imread(imagefile)
    dispsize = img.transpose(1, 0).shape

    fig, ax = draw_display(dispsize, imagefile=imagefile)

    if durationsize:
        siz = fix['dur'] * (10.0**(4.5))
    else:
        siz = 1 * np.median(fix['dur']/30.0)*1000

    if durationcolour:
        col = fix['dur']
    else:
        col = COLS['chameleon'][2]

    ax.scatter(fix['x'], fix['y'], s=siz, c=col, marker='o',
               cmap='jet', alpha=alpha, edgecolors='none')
    ax.invert_yaxis()

    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_scanpath(fix, imagefile=None, alpha=0.5, savefilename=None,  durationsize=True, durationcolour=True,):
    """Draws a scanpath: a series of arrows between numbered fixations,
    optionally drawn over an image

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                                    as produced by edfreader.read_edf, e.g.
                                    edfdata[trialnr]['events']['Efix']
    saccades		-	a list of saccade ending events from a single trial,
                                    as produced by edfreader.read_edf, e.g.
                                    edfdata[trialnr]['events']['Esac']
    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                                    the heatmap, where 0 is completely transparant and 1
                                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                                    heatmap
    """

    img = image.imread(imagefile)
    dispsize = img.transpose(1, 0).shape

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    if durationsize:
        siz = fix['dur'] * (10.0**(4.5))
    else:
        siz = 1 * np.median(fix['dur']/30.0)*1000

    if durationcolour:
        col = fix['dur']
    else:
        col = COLS['chameleon'][2]

    ax.scatter(fix['x'], fix['y'], s=siz, c=col, marker='o',
               cmap='jet', alpha=alpha, edgecolors='none')

    # draw fixations
    # ax.scatter(fix['x'],fix['y'], s=(fix['dur'] * (10.0**(4.5))), c=fix['dur'], marker='o', cmap='jet', alpha=alpha, edgecolors='none')

    # draw annotations (fixation numbers)
    for i in range(len(fix['x'])):
        ax.annotate(str(i+1), (fix['x'][i], fix['y'][i]), color=COLS['aluminium'][5], alpha=1,
                    horizontalalignment='center', verticalalignment='center', multialignment='center', fontsize=fix['dur'][i] * 55)

    # loop through all saccades
    for x, y, dx, dy in zip(fix['x'], fix['y'], fix['dx'], fix['dy']):
        # draw an arrow between every saccade start and ending
        ax.arrow(x, y, dx, dy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5],
                 fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x/2
    yo = y/2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)
                                      ) + ((float(j)-yo)**2/(2*sy*sy))))

    return M


def draw_heatmap(fix, imagefile=None, alpha=0.5, savefilename=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                                    as produced by edfreader.read_edf, e.g.
                                    edfdata[trialnr]['events']['Efix']
    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)
    durationweight	-	Boolean indicating whether the fixation duration is
                                    to be taken into account as a weight for the heatmap
                                    intensity; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                                    the heatmap, where 0 is completely transparant and 1
                                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                                    heatmap
    """

    img = image.imread(imagefile)
    dispsize = img.transpose(1, 0).shape

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh/2)
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(fix['dur'])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = strt + int(fix['x'][i]) - int(gwh/2)
        y = strt + int(fix['y'][i]) - int(gwh/2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1], x:x+hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh, x:x+gwh] += gaus * fix['dur'][i]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1]+strt, strt:dispsize[0]+strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def draw_raw(x, y, imagefile=None, savefilename=None, markersize=10):

    img = image.imread(imagefile)
    dispsize = img.transpose(1, 0).shape

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # plot raw data points
    ax.plot(x, y, 'o', color=COLS['aluminium'][0], markeredgecolor=COLS['aluminium'][5], markersize=markersize)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig

def get_fixations_dict_from_reflacx_eye_tracking(relfacx_eye_tracking_df):
    relfacx_eye_tracking_df['x'] = relfacx_eye_tracking_df['x_position'] 
    relfacx_eye_tracking_df['y'] = relfacx_eye_tracking_df['y_position'] 
    relfacx_eye_tracking_df['duration']=relfacx_eye_tracking_df['timestamp_end_fixation'] - relfacx_eye_tracking_df['timestamp_start_fixation'] 

    # make saccade
    relfacx_eye_tracking_df['saccade'] = None
    for i in range(len(relfacx_eye_tracking_df)-1):
        relfacx_eye_tracking_df.loc[i+1, 'dx'] = relfacx_eye_tracking_df.loc[i+1, "x"] - relfacx_eye_tracking_df.loc[i, "x"]
        relfacx_eye_tracking_df.loc[i+1, 'dy'] = relfacx_eye_tracking_df.loc[i+1, "y"] - relfacx_eye_tracking_df.loc[i, "y"]

    return { 
        'x': np.array(relfacx_eye_tracking_df['x']),
        'y': np.array(relfacx_eye_tracking_df['y']),
        'dur': np.array(relfacx_eye_tracking_df['duration']),
        'dx': np.array(relfacx_eye_tracking_df['dx'][1:]),
        'dy': np.array(relfacx_eye_tracking_df['dy'][1:])
    }


def get_fixations_dict_from_eyegaze_eye_tracking(eye_gaze_eyetracking_df):
    eye_gaze_eyetracking_df['x'] = eye_gaze_eyetracking_df['X_ORIGINAL'] 
    eye_gaze_eyetracking_df['y'] = eye_gaze_eyetracking_df['Y_ORIGINAL'] 
    eye_gaze_eyetracking_df['time']=eye_gaze_eyetracking_df['Time (in secs)'] 

    # make saccade
    eye_gaze_eyetracking_df['saccade'] = None
    for i in range(len(eye_gaze_eyetracking_df)-1):
        eye_gaze_eyetracking_df.loc[i+1,'duration'] = eye_gaze_eyetracking_df.loc[i+1, "time"] - eye_gaze_eyetracking_df.loc[i, "time"]
        eye_gaze_eyetracking_df.loc[i+1, 'dx'] = eye_gaze_eyetracking_df.loc[i+1, "x"] - eye_gaze_eyetracking_df.loc[i, "x"]
        eye_gaze_eyetracking_df.loc[i+1, 'dy'] = eye_gaze_eyetracking_df.loc[i+1, "y"] - eye_gaze_eyetracking_df.loc[i, "y"]

    return { 
        'x': np.array(eye_gaze_eyetracking_df['x']),
        'y': np.array(eye_gaze_eyetracking_df['y']),
        'dur': np.array(eye_gaze_eyetracking_df['duration']),
        'dx': np.array(eye_gaze_eyetracking_df['dx'][1:]),
        'dy': np.array(eye_gaze_eyetracking_df['dy'][1:])
    }