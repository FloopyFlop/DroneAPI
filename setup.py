from setuptools import find_packages, setup

package_name = 'drone_api'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='j10pr',
    maintainer_email='j10pr@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_demo = drone_api.test_demo:main",
            "test_stream_logging = drone_api.test_stream_logging:main"
        ],
    },
)
