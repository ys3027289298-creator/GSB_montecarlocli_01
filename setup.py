from setuptools import setup

setup(
    name='montecarlocli',
    version='0.5',
    py_modules=['montecarlocli'],
    install_requires=[],
    entry_points='''
    [console_scripts]
    montecarlocli=montecarlocli:main
    '''
)
