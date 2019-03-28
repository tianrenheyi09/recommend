# # cross_part
import tensorflow as tf
cross_layer_num = 1
total_size = 2
x0 = tf.reshape(tf.constant([1,2,3,4,5,6],dtype=tf.float32),shape=[3,2])
weights = tf.reshape(tf.constant([1,1,1,1],dtype=tf.float32),shape=[2,2])
bs = tf.reshape(tf.constant([1,1,1,1],dtype=tf.float32),shape=[2,2])

sess = tf.Session()

_x0 = tf.reshape(x0, (-1, total_size, 1))
x_l = _x0

aa = tf.tensordot(tf.matmul(_x0, x_l, transpose_b=True),
                        tf.reshape(weights[0],shape=[2,1]),1)
bb = aa+tf.reshape(bs[0],[2,1])
cc = bb+x_l

for l in range(cross_layer_num):
    x_l = tf.tensordot(tf.matmul(_x0, x_l, transpose_b=True),
                        tf.reshape(weights[l],shape=[2,1]),1) +tf.reshape(bs[l],[2,1]) + x_l



print('cross layer is lllll')
cross_network_out = tf.reshape(x_l, (-1, total_size))


sess.run(cross_network_out)


m_x0 = x0
m_x_l = m_x0

for i in range(cross_layer_num):
    xlw = tf.matmul(m_x_l,tf.reshape(weights[l],shape=[2,1]))
    m_x_l = x0*xlw+m_x_l+tf.reshape(bs[l],[1,2])

sess.run(m_x_l)
####my cross net
# self._x0 = tf.reshape(self.x0, (-1, self.total_size))
# x_l = self._x0
# for i in range(self.cross_layer_num):
#     xlw = tf.matmul(x_l,self.weights['cross_layer_%d'%i])
#     x_l = self._x0*xlw+x_l+self.weights["cross_bias_%d" %i]
#
# self.cross_network_out = tf.reshape(x_l, (-1, self.total_size))