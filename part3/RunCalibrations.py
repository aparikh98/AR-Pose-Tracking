import CalibrationHelpers as calib
intrinsics, distortion, roi, new_intrinsics = calib.CalibrateCamera('calibration_data', True)

calib.SaveCalibrationData('calibration_data', intrinsics, distortion, new_intrinsics,
                        roi)
print(calib.LoadCalibrationData('calibration_data'))
