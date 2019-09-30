from setuptools import setup

package_name = 'segmentation_publisher_v2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'segmentation_pub_test',
        'segmentation_test_v2', 
        'lednet', 
        'transform'
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    author='yourname',
    author_email='email@hoge',
    maintainer='yourname',
    maintainer_email='email@hoge',
    keywords=['ROS', 'ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Examples of minimal publishers using rclpy.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_pub_test = segmentation_pub_test:main',
            'segmentation_pub_v2 = segmentation_test_v2:main',
        ],
    },
)
