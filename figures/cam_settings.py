

def set_cam_settings(viewer, settings_dict):
    viewer.cam.distance = settings_dict['distance']
    viewer.cam.lookat[0] = settings_dict['lookat'][0]
    viewer.cam.lookat[1] = settings_dict['lookat'][1]
    viewer.cam.lookat[2] = settings_dict['lookat'][2]
    viewer.cam.elevation = settings_dict['elevation']
    viewer.cam.azimuth = settings_dict['azimuth']


CAM_SETTINGS = {
    'sawyer_drawer_open': {
        'distance': 0.719,
        'lookat': [-0.008, 0.583, 0.153],
        'elevation': -29.861,
        'azimuth': 159.408,
    },
    'sawyer_drawer_close': {
        'distance': 0.719,
        'lookat': [-0.008, 0.583, 0.153],
        'elevation': -29.861,
        'azimuth': 159.408,
    },
    'sawyer_push': {
        'distance': 1.107,
        'lookat': [-0.099, 0.558, 0.104],
        'elevation': -24.344,
        'azimuth': -178.653,
    },
    'sawyer_lift': {
        'distance': 1.107,
        'lookat': [-0.099, 0.558, 0.104],
        'elevation': -24.344,
        'azimuth': -178.653,
    },
    'sawyer_box_close': {
        'distance': 0.971,
        'lookat': [-0.077, 0.625, 0.108],
        'elevation': -14.850,
        'azimuth': -163.001,
    },
    'sawyer_bin_picking': {
        'distance': 0.867,
        'lookat': [0.046, 0.524, 0.091],
        'elevation': -24.344,
        'azimuth': -87.562,
    },
    'door-human-v0': {
        'distance': 1.087,
        'lookat': [0.02996559, -0.15030982, 0.30230237],
        'elevation': -45.000,
        'azimuth': 90.000,
    },
    'door-human-v0-dp': {
        'distance': 1.087,
        'lookat': [0.02996559, -0.15030982, 0.30230237],
        'elevation': -45.000,
        'azimuth': 90.000,
    },
    'hammer-human-v0': {
        'distance': 0.639,
        'lookat': [-0.133, -0.230, 0.239],
        'elevation': -47.951,
        'azimuth': 27.263,
    },
    'hammer-human-v0-dp': {
        'distance': 0.639,
        'lookat': [-0.133, -0.230, 0.239],
        'elevation': -47.951,
        'azimuth': 27.263,
    },
    'relocate-human-v0': {
        'distance': 0.880,
        'lookat': [0.122, 0.020, 0.165],
        'elevation': -26.140,
        'azimuth': 144.269,
    },
    'relocate-human-v0-najp-dp': {
        'distance': 0.880,
        'lookat': [0.122, 0.020, 0.165],
        'elevation': -26.140,
        'azimuth': 144.269,
    },
}