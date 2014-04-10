__author__ = "Ian Goodfellow"

def cloud_to_ppf(x, y, higher_better=True):
    """
    x: A list of x coordinates
    y: A list of y coordinates

    Makes a production possibility frontier, i.e. a curve showing
    how much y is possible for each value of x.

    If higher_better, we assume bigger values of y are better, so we
    make a curve that lies above all points in the cloud.

    Returns ppf_x, ppf_y, which are lists of x and y points defining the ppf
    curve.
    """

    print 'running cloud to ppf'

    combined = zip(x, y)
    s = sorted(combined, key = lambda e : e[0])
    x = [e[0] for e in s]
    y = [e[1] for e in s]

    i = 0
    while i < len(x):
        lx = x[i]
        ly = y[i]
        j = len(x) - 1
        while j > i:
            rx = x[j]
            #assert rx > lx
            ry = y[j]
            slope = (ry - ly) / (rx - lx)
            intercept = ly - slope * lx
            k = i + 1

            while k < j:
                mx = x[k]
                my = y[k]
                pred = slope * mx + intercept
                if (higher_better and my <= pred) or (my >= pred and not
                        higher_better):
                    del x[k]
                    del y[k]
                    j -= 1
                else:
                    k += 1
            j -= 1
        i += 1



    print 'done running cloud to pff'

    return x, y


