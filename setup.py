from setuptools import setup

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml',
                                   'launch/object_detection.launch.xml']),
        ('lib/' + package_name + '/models/',['models/experimental.py', 'models/common.py',
                                             'models/yolo.py']),
        ('lib/' + package_name + '/utils/',['utils/general.py', 'utils/torch_utils.py',
                                             'utils/plots.py', 'utils/datasets.py',
                                             'utils/google_utils.py', 'utils/activations.py',
                                             'utils/add_nms.py', 'utils/autoanchor.py',
                                             'utils/loss.py', 'utils/metrics.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marno Nel & rshima',
    maintainer_email='marthinusnel2023@u.northwestern.edu & rintarohshima2023@u.northwestern.edu',
    description='A ROS 2 package to deploy YOLOv7 trained models on',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_deli_flir = object_detection.deliver.yolo_deli_flir:main',
            'yolo_deli_logi_direct = object_detection.deliver.yolo_deli_logi_direct:main',
            'yolo_deli_logi_sub = object_detection.deliver.yolo_deli_logi_sub:main',
            'yolo_amz = object_detection.amz.yolo_amz:main',
            'yolo_amz_high = object_detection.amz.yolo_amz_high:main',
            'yolo_amz_high_sync = object_detection.amz.yolo_amz_high_sync:main',
            'yolo_amz_sub = object_detection.amz.yolo_amz_sub:main',
            'yolo_amz_high_sync_rightup = object_detection.amz.yolo_amz_high_sync_rightup:main'
        ],
    },
)
