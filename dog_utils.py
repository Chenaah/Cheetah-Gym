import numpy as np
import math

PI = 3.14159


def box(array, tuple_list):
	reach_limit = False
	def box_element(x, min_x, max_x):
		nonlocal reach_limit
		if x < min_x:
			reach_limit = True
			return min_x
		elif x > max_x:
			reach_limit = True
			return max_x
		else:
			return x
	return np.array([box_element(array[i], tuple_list[i][0], tuple_list[i][1]) for i in range(np.size(array))]), reach_limit


def quat_to_YZX(quat):

	def compose( position, quaternion, scale ):
		te = []
		x = quaternion[0]
		y = quaternion[1]
		z = quaternion[2]
		w = quaternion[3]
		x2 = x + x
		y2 = y + y
		z2 = z + z
		xx = x * x2
		xy = x * y2
		xz = x * z2
		yy = y * y2
		yz = y * z2
		zz = z * z2
		wx = w * x2
		wy = w * y2
		wz = w * z2
		sx = scale[0]
		sy = scale[1]
		sz = scale[2]
		te.append(( 1 - ( yy + zz ) ) * sx)
		te.append(( xy + wz ) * sx)
		te.append(( xz - wy ) * sx)
		te.append(0)
		te.append(( xy - wz ) * sy)
		te.append(( 1 - ( xx + zz ) ) * sy)
		te.append(( yz + wx ) * sy)
		te.append(0)
		te.append(( xz + wy ) * sz)
		te.append(( yz - wx ) * sz)
		te.append( ( 1 - ( xx + yy ) ) * sz)
		te.append(0)
		te.append( position[0])
		te.append( position[1])
		te.append( position[2])
		te.append(1)

		return te

	mat = compose([0]*3, quat, [1]*3)
	m11 = mat[0]
	m12 = mat[4]
	m13 = mat[8]
	m21 = mat[1]
	m22 = mat[5]
	m23 = mat[9]
	m31 = mat[2]
	m32 = mat[6]
	m33 = mat[1]
	_z = math.asin(max(min(m21, 1), -1));

	if abs(m21) < 0.9999999:
		_x = math.atan2(-m23, m22);
		_y = math.atan2(-m31, m11);
	else:
		_x = 0;
		_y = math.atan2(m13, m33);

	return [_x, _y, _z]