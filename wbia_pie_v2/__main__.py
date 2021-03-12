# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_pie_v2

    print('Looks like the imports worked')
    print('wbia_pie_v2 = {!r}'.format(wbia_pie_v2))
    print('wbia_pie_v2.__file__ = {!r}'.format(wbia_pie_v2.__file__))
    print('wbia_pie_v2.__version__ = {!r}'.format(wbia_pie_v2.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m wbia_pie_v2
    """
    main()
