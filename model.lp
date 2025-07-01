\ Model transfer_tensor_alloc
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  total_latency
Subject To
 one_path_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1']):
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   = 1
 path_core_link_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_Core_2:
   - x_conv1.W_Core_2
   + y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   <= 0
 path_core_link_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_Core_6:
   - x_conv1.W_Core_6
   + y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   <= 0
 path_core_link_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_Core_1:
   - x_conv1.W_Core_1
   + y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   <= 0
 one_path_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1']):
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   = 1
 path_core_link_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_Core_2:
   - x_conv1.I_Core_2
   + y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   <= 0
 path_core_link_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_Core_6:
   - x_conv1.I_Core_6
   + y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   <= 0
 path_core_link_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_Core_1:
   - x_conv1.I_Core_1
   + y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   <= 0
 one_path_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1']):
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   + y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   + y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   = 1
 path_core_link_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_Core_3:
   - x_conv2.W_Core_3
   + y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   <= 0
 path_core_link_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_Core_6:
   - x_conv2.W_Core_6
   + y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   <= 0
 path_core_link_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_Core_0:
   - x_conv2.W_Core_0
   + y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   <= 0
 one_path_Transfer(conv2.part0.O_->_['conv2.O']):
   y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683
   + y_Transfer(conv2.part0.O_->_['conv2.O'])_5740354900026072187
   + y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497 = 1
 one_path_Transfer(conv2.part1.O_->_['conv2.O']):
   y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481
   + y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683
   + y_Transfer(conv2.part1.O_->_['conv2.O'])_5740354900026072187 = 1
 one_path_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1']):
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   = 1
 one_path_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1']):
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   = 1
 place_conv1.W: x_conv1.W_Core_6 + x_conv1.W_Core_1 + x_conv1.W_Core_2 = 1
 place_conv1.I: x_conv1.I_Core_6 + x_conv1.I_Core_1 + x_conv1.I_Core_2 = 1
 place_conv2.W: x_conv2.W_Core_6 + x_conv2.W_Core_0 + x_conv2.W_Core_3 = 1
 place_conv2.O: x_conv2.O_Core_6 + x_conv2.O_Core_0 + x_conv2.O_Core_3 = 1
 link_usage_CommunicationLink(Core(2),_Core(1),_bw=32)_5:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   <= 1
 link_usage_CommunicationLink(Core(6),_Any,_bw=64)_5:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   <= 1
 link_usage_CommunicationLink(Core(1),_Core(2),_bw=32)_5:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   <= 1
 link_usage_CommunicationLink(Core(2),_Core(1),_bw=32)_6:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   <= 1
 link_usage_CommunicationLink(Core(6),_Any,_bw=64)_6:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   <= 1
 link_usage_CommunicationLink(Core(1),_Core(2),_bw=32)_6:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   <= 1
 link_usage_CommunicationLink(Core(3),_Core(0),_bw=32)_7:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   <= 1
 link_usage_CommunicationLink(Core(6),_Any,_bw=64)_7:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   <= 1
 link_usage_CommunicationLink(Core(0),_Core(3),_bw=32)_7:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   <= 1
 link_usage_CommunicationLink(Core(6),_Any,_bw=64)_20:
   y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683 <= 1
 link_usage_CommunicationLink(Core(0),_Core(3),_bw=32)_20:
   y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497 <= 1
 link_usage_CommunicationLink(Core(3),_Core(0),_bw=32)_21:
   y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481 <= 1
 link_usage_CommunicationLink(Core(6),_Any,_bw=64)_21:
   y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683 <= 1
 link_usage_CommunicationLink(Core(1),_Core(0),_bw=32)_13:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   <= 1
 link_usage_CommunicationLink(Core(0),_Core(3),_bw=32)_13:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   <= 1
 link_usage_CommunicationLink(Core(2),_Core(3),_bw=32)_14:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   <= 1
 link_usage_CommunicationLink(Core(3),_Core(0),_bw=32)_14:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   <= 1
 mem_cap_Core_0: 4608 x_conv2.W_Core_0 + 3584 x_conv2.O_Core_0
   <= 1.6760064e+07
 mem_cap_Core_3: 4608 x_conv2.W_Core_3 + 3584 x_conv2.O_Core_3
   <= 1.6760064e+07
 mem_cap_Core_1: 4608 x_conv1.W_Core_1 + 11136 x_conv1.I_Core_1
   <= 1.675968e+07
 mem_cap_Core_2: 4608 x_conv1.W_Core_2 + 11136 x_conv1.I_Core_2
   <= 1.675968e+07
 mem_cap_Core_6: 4608 x_conv1.W_Core_6 + 11136 x_conv1.I_Core_6
   + 4608 x_conv2.W_Core_6 + 3584 x_conv2.O_Core_6 <= 1e+10
 ssc_lat_conv2.part0: L_17 >= 513
 ssc_lat_conv2.part1: L_17 >= 513
 ssc_lat_conv1.part0: L_10 >= 513
 ssc_lat_conv1.part1: L_10 >= 513
 tr_lat_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015:
   - 144 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + L_5 >= 0
 tr_lat_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683:
   - 72 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + L_5 >= 0
 tr_lat_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689:
   - 144 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + L_5 >= 0
 tr_lat_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015:
   - 348 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + L_6 >= 0
 tr_lat_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683:
   - 174 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + L_6 >= 0
 tr_lat_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689:
   - 348 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + L_6 >= 0
 tr_lat_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481:
   - 144 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   + L_7 >= 0
 tr_lat_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683:
   - 72 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   + L_7 >= 0
 tr_lat_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497:
   - 144 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   + L_7 >= 0
 tr_lat_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683:
   - 28 y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683 + L_20
   >= 0
 tr_lat_Transfer(conv2.part0.O_->_['conv2.O'])_5740354900026072187: L_20
   >= 0
 tr_lat_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497:
   - 56 y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497
   + L_20 >= 0
 tr_lat_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481:
   - 56 y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481 + L_21
   >= 0
 tr_lat_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683:
   - 28 y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683 + L_21
   >= 0
 tr_lat_Transfer(conv2.part1.O_->_['conv2.O'])_5740354900026072187: L_21
   >= 0
 tr_lat_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927:
   - 56 y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   + L_13 >= 0
 tr_lat_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869:
   - 56 y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   + L_14 >= 0
 R63: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_0 <= 36
 R64: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_0 >= 1
 R65: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_1 <= 36
 R66: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_1 >= 1
 R67: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_2 <= 36
 R68: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_2 >= 1
 R69: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_3 <= 36
 R70: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_3 >= 1
 R71: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_4 <= 36
 R72: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_4 >= 1
 R73: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_5 <= 36
 R74: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_5 >= 1
 R75: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_6 <= 36
 R76: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_6 >= 1
 R77:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_7 <= 36
 R78:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_7 >= 1
 R79: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_8 <= 36
 R80: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_8 >= 1
 R81: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_9 <= 36
 R82: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_9 >= 1
 R83: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_10 <= 36
 R84: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_10 >= 1
 R85: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_11 <= 36
 R86: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_11 >= 1
 R87: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_12 <= 36
 R88: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_12 >= 1
 R89: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_13 <= 36
 R90: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_13 >= 1
 R91:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_14 <= 36
 R92:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_14 >= 1
 R93: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_15 <= 36
 R94: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_15 >= 1
 R95: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_16 <= 36
 R96: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_16 >= 1
 R97: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_17 <= 36
 R98: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_17 >= 1
 R99: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_18 <= 36
 R100: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_18 >= 1
 R101: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_19 <= 36
 R102: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_19 >= 1
 R103: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_20 <= 36
 R104: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_20 >= 1
 R105: y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_21 <= 36
 R106: y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481
   + 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_21 >= 1
 R107: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_22 <= 36
 R108: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_22 >= 1
 R109: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_23 <= 36
 R110: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_23 >= 1
 R111: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_24 <= 36
 R112: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_24 >= 1
 R113: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_25 <= 36
 R114: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_25 >= 1
 R115: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_26 <= 36
 R116: 36 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_26 >= 1
 R117: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_0 <= 36
 R118: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_0 >= 1
 R119: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_1 <= 36
 R120: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_1 >= 1
 R121: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_2 <= 36
 R122: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_2 >= 1
 R123: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_3 <= 36
 R124: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_3 >= 1
 R125: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_4 <= 36
 R126: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_4 >= 1
 R127:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_5 <= 36
 R128:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_5 >= 1
 R129:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_6 <= 36
 R130:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
   + 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_6 >= 1
 R131: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_7 <= 36
 R132: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_7 >= 1
 R133: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_8 <= 36
 R134: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_8 >= 1
 R135: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_9 <= 36
 R136: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_9 >= 1
 R137: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_10 <= 36
 R138: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_10 >= 1
 R139: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_11 <= 36
 R140: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_11 >= 1
 R141: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_12 <= 36
 R142: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_12 >= 1
 R143: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_13 <= 36
 R144: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_13 >= 1
 R145: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_14 <= 36
 R146: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_14 >= 1
 R147: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_15 <= 36
 R148: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_15 >= 1
 R149: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_16 <= 36
 R150: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_16 >= 1
 R151: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_17 <= 36
 R152: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_17 >= 1
 R153: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_18 <= 36
 R154: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_18 >= 1
 R155: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_19 <= 36
 R156: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_19 >= 1
 R157: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_20 <= 36
 R158: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_20 >= 1
 R159: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_21 <= 36
 R160: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_21 >= 1
 R161: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_22 <= 36
 R162: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_22 >= 1
 R163: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_23 <= 36
 R164: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_23 >= 1
 R165: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_24 <= 36
 R166: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_24 >= 1
 R167: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_25 <= 36
 R168: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_25 >= 1
 R169: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_26 <= 36
 R170: 36 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_26 >= 1
 R171: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_0 <= 36
 R172: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_0 >= 1
 R173: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_1 <= 36
 R174: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_1 >= 1
 R175: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_2 <= 36
 R176: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_2 >= 1
 R177: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_3 <= 36
 R178: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_3 >= 1
 R179: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_4 <= 36
 R180: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_4 >= 1
 R181: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_5 <= 36
 R182: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_5 >= 1
 R183: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_6 <= 36
 R184: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_6 >= 1
 R185: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_7 <= 36
 R186: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_7 >= 1
 R187: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_8 <= 36
 R188: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_8 >= 1
 R189: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_9 <= 36
 R190: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_9 >= 1
 R191: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_10 <= 36
 R192: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_10 >= 1
 R193: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_11 <= 36
 R194: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_11 >= 1
 R195: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_12 <= 36
 R196: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_12 >= 1
 R197:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   + 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_13 <= 36
 R198:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   + 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_13 >= 1
 R199: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_14 <= 36
 R200: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_14 >= 1
 R201: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_15 <= 36
 R202: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_15 >= 1
 R203: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_16 <= 36
 R204: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_16 >= 1
 R205: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_17 <= 36
 R206: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_17 >= 1
 R207: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_18 <= 36
 R208: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_18 >= 1
 R209: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_19 <= 36
 R210: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_19 >= 1
 R211: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_20 <= 36
 R212: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_20 >= 1
 R213: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_21 <= 36
 R214: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_21 >= 1
 R215: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_22 <= 36
 R216: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_22 >= 1
 R217: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_23 <= 36
 R218: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_23 >= 1
 R219: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_24 <= 36
 R220: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_24 >= 1
 R221: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_25 <= 36
 R222: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_25 >= 1
 R223: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_26 <= 36
 R224: 36 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_26 >= 1
 R225: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_0 <= 36
 R226: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_0 >= 1
 R227: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_1 <= 36
 R228: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_1 >= 1
 R229: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_2 <= 36
 R230: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_2 >= 1
 R231: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_3 <= 36
 R232: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_3 >= 1
 R233: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_4 <= 36
 R234: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_4 >= 1
 R235:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_5 <= 36
 R236:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_5 >= 1
 R237:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_6 <= 36
 R238:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
   + 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_6 >= 1
 R239: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_7 <= 36
 R240: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_7 >= 1
 R241: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_8 <= 36
 R242: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_8 >= 1
 R243: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_9 <= 36
 R244: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_9 >= 1
 R245: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_10 <= 36
 R246: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_10 >= 1
 R247: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_11 <= 36
 R248: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_11 >= 1
 R249: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_12 <= 36
 R250: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_12 >= 1
 R251: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_13 <= 36
 R252: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_13 >= 1
 R253: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_14 <= 36
 R254: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_14 >= 1
 R255: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_15 <= 36
 R256: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_15 >= 1
 R257: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_16 <= 36
 R258: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_16 >= 1
 R259: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_17 <= 36
 R260: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_17 >= 1
 R261: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_18 <= 36
 R262: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_18 >= 1
 R263: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_19 <= 36
 R264: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_19 >= 1
 R265: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_20 <= 36
 R266: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_20 >= 1
 R267: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_21 <= 36
 R268: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_21 >= 1
 R269: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_22 <= 36
 R270: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_22 >= 1
 R271: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_23 <= 36
 R272: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_23 >= 1
 R273: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_24 <= 36
 R274: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_24 >= 1
 R275: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_25 <= 36
 R276: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_25 >= 1
 R277: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_26 <= 36
 R278: 36 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_26 >= 1
 R279: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_0 <= 36
 R280: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_0 >= 1
 R281: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_1 <= 36
 R282: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_1 >= 1
 R283: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_2 <= 36
 R284: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_2 >= 1
 R285: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_3 <= 36
 R286: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_3 >= 1
 R287: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_4 <= 36
 R288: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_4 >= 1
 R289:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_5 <= 36
 R290:
   y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_5 >= 1
 R291:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_6 <= 36
 R292:
   y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_6 >= 1
 R293:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_7 <= 36
 R294:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_7 >= 1
 R295: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_8 <= 36
 R296: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_8 >= 1
 R297: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_9 <= 36
 R298: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_9 >= 1
 R299: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_10 <= 36
 R300: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_10 >= 1
 R301: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_11 <= 36
 R302: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_11 >= 1
 R303: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_12 <= 36
 R304: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_12 >= 1
 R305: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_13 <= 36
 R306: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_13 >= 1
 R307: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_14 <= 36
 R308: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_14 >= 1
 R309: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_15 <= 36
 R310: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_15 >= 1
 R311: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_16 <= 36
 R312: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_16 >= 1
 R313: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_17 <= 36
 R314: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_17 >= 1
 R315: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_18 <= 36
 R316: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_18 >= 1
 R317: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_19 <= 36
 R318: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_19 >= 1
 R319: y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_20 <= 36
 R320: y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_20 >= 1
 R321: y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_21 <= 36
 R322: y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683
   + 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_21 >= 1
 R323: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_22 <= 36
 R324: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_22 >= 1
 R325: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_23 <= 36
 R326: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_23 >= 1
 R327: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_24 <= 36
 R328: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_24 >= 1
 R329: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_25 <= 36
 R330: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_25 >= 1
 R331: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_26 <= 36
 R332: 36 idleS_CommunicationLink(Core(6),_Any,_bw=64)_26 >= 1
 R333: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_0 <= 36
 R334: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_0 >= 1
 R335: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_1 <= 36
 R336: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_1 >= 1
 R337: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_2 <= 36
 R338: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_2 >= 1
 R339: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_3 <= 36
 R340: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_3 >= 1
 R341: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_4 <= 36
 R342: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_4 >= 1
 R343: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_5 <= 36
 R344: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_5 >= 1
 R345: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_6 <= 36
 R346: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_6 >= 1
 R347:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_7 <= 36
 R348:
   y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_7 >= 1
 R349: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_8 <= 36
 R350: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_8 >= 1
 R351: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_9 <= 36
 R352: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_9 >= 1
 R353: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_10 <= 36
 R354: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_10 >= 1
 R355: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_11 <= 36
 R356: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_11 >= 1
 R357: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_12 <= 36
 R358: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_12 >= 1
 R359:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_13 <= 36
 R360:
   y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_13 >= 1
 R361: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_14 <= 36
 R362: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_14 >= 1
 R363: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_15 <= 36
 R364: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_15 >= 1
 R365: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_16 <= 36
 R366: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_16 >= 1
 R367: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_17 <= 36
 R368: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_17 >= 1
 R369: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_18 <= 36
 R370: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_18 >= 1
 R371: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_19 <= 36
 R372: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_19 >= 1
 R373: y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_20 <= 36
 R374: y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497
   + 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_20 >= 1
 R375: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_21 <= 36
 R376: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_21 >= 1
 R377: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_22 <= 36
 R378: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_22 >= 1
 R379: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_23 <= 36
 R380: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_23 >= 1
 R381: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_24 <= 36
 R382: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_24 >= 1
 R383: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_25 <= 36
 R384: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_25 >= 1
 R385: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_26 <= 36
 R386: 36 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_26 >= 1
 R387: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_0 <= 36
 R388: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_0 >= 1
 R389: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_1 <= 36
 R390: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_1 >= 1
 R391: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_2 <= 36
 R392: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_2 >= 1
 R393: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_3 <= 36
 R394: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_3 >= 1
 R395: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_4 <= 36
 R396: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_4 >= 1
 R397: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_5 <= 36
 R398: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_5 >= 1
 R399: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_6 <= 36
 R400: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_6 >= 1
 R401: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_7 <= 36
 R402: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_7 >= 1
 R403: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_8 <= 36
 R404: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_8 >= 1
 R405: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_9 <= 36
 R406: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_9 >= 1
 R407: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_10 <= 36
 R408: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_10 >= 1
 R409: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_11 <= 36
 R410: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_11 >= 1
 R411: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_12 <= 36
 R412: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_12 >= 1
 R413: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_13 <= 36
 R414: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_13 >= 1
 R415:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   + 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_14 <= 36
 R416:
   y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
   + 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_14 >= 1
 R417: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_15 <= 36
 R418: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_15 >= 1
 R419: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_16 <= 36
 R420: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_16 >= 1
 R421: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_17 <= 36
 R422: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_17 >= 1
 R423: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_18 <= 36
 R424: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_18 >= 1
 R425: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_19 <= 36
 R426: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_19 >= 1
 R427: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_20 <= 36
 R428: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_20 >= 1
 R429: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_21 <= 36
 R430: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_21 >= 1
 R431: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_22 <= 36
 R432: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_22 >= 1
 R433: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_23 <= 36
 R434: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_23 >= 1
 R435: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_24 <= 36
 R436: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_24 >= 1
 R437: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_25 <= 36
 R438: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_25 >= 1
 R439: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_26 <= 36
 R440: 36 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_26 >= 1
 overlap_le_Core_0: overlap <= 1e-06
 overlap_le_Core_1: overlap <= 1e-06
 overlap_le_Core_2: overlap <= 1e-06
 overlap_le_Core_3: overlap <= 1e-06
 overlap_le_None: overlap <= 1e-06
 total_latency_def: - 54 L_0 - 54 L_1 - 54 L_2 - 54 L_3 - 54 L_4 - 54 L_5
   - 54 L_6 - 54 L_7 - 54 L_8 - 54 L_9 - 54 L_10 - 54 L_11 - 54 L_12
   - 54 L_13 - 54 L_14 - 54 L_15 - 54 L_16 - 54 L_17 - 54 L_18 - 54 L_19
   - 54 L_20 - 54 L_21 - 54 L_22 - 54 L_23 - 54 L_24 - 54 L_25 - 54 L_26
   + 53 overlap + total_latency = 0
 overlap_le_CommunicationLink(Core(3),_Core(0),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(1),_Core(2),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(1),_Core(0),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(2),_Core(1),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(6),_Any,_bw=64): overlap + [
   - L_0 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_0
   - L_0 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_0
   - L_1 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_1
   - L_1 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_1
   - L_2 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_2
   - L_2 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_2
   - L_3 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_3
   - L_3 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_3
   - L_4 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_4
   - L_4 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_4
   - L_5 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_5
   - L_5 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_5
   - L_6 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_6
   - L_6 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_6
   - L_7 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_7
   - L_7 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_7
   - L_8 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_8
   - L_8 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_8
   - L_9 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_9
   - L_9 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_9
   - L_10 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_10
   - L_10 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_10
   - L_11 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_11
   - L_11 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_11
   - L_12 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_12
   - L_12 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_12
   - L_13 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_13
   - L_13 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_13
   - L_14 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_14
   - L_14 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_14
   - L_15 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_15
   - L_15 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_15
   - L_16 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_16
   - L_16 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_16
   - L_17 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_17
   - L_17 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_17
   - L_18 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_18
   - L_18 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_18
   - L_19 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_19
   - L_19 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_19
   - L_20 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_20
   - L_20 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_20
   - L_21 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_21
   - L_21 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_21
   - L_22 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_22
   - L_22 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_22
   - L_23 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_23
   - L_23 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_23
   - L_24 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_24
   - L_24 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_24
   - L_25 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_25
   - L_25 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_25
   - L_26 * idleS_CommunicationLink(Core(6),_Any,_bw=64)_26
   - L_26 * idleE_CommunicationLink(Core(6),_Any,_bw=64)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(0),_Core(3),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_26 ] <= 1e-06
 overlap_le_CommunicationLink(Core(2),_Core(3),_bw=32): overlap + [
   - L_0 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_0
   - L_0 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_0
   - L_1 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_1
   - L_1 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_1
   - L_2 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_2
   - L_2 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_2
   - L_3 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_3
   - L_3 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_3
   - L_4 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_4
   - L_4 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_4
   - L_5 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_5
   - L_5 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_5
   - L_6 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_6
   - L_6 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_6
   - L_7 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_7
   - L_7 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_7
   - L_8 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_8
   - L_8 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_8
   - L_9 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_9
   - L_9 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_9
   - L_10 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_10
   - L_10 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_10
   - L_11 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_11
   - L_11 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_11
   - L_12 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_12
   - L_12 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_12
   - L_13 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_13
   - L_13 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_13
   - L_14 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_14
   - L_14 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_14
   - L_15 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_15
   - L_15 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_15
   - L_16 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_16
   - L_16 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_16
   - L_17 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_17
   - L_17 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_17
   - L_18 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_18
   - L_18 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_18
   - L_19 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_19
   - L_19 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_19
   - L_20 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_20
   - L_20 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_20
   - L_21 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_21
   - L_21 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_21
   - L_22 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_22
   - L_22 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_22
   - L_23 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_23
   - L_23 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_23
   - L_24 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_24
   - L_24 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_24
   - L_25 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_25
   - L_25 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_25
   - L_26 * idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_26
   - L_26 * idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_26 ] <= 1e-06
Bounds
Binaries
 x_conv1.W_Core_6 x_conv1.W_Core_1 x_conv1.W_Core_2 x_conv1.I_Core_6
 x_conv1.I_Core_1 x_conv1.I_Core_2 x_conv2.W_Core_6 x_conv2.W_Core_0
 x_conv2.W_Core_3 x_conv2.O_Core_6 x_conv2.O_Core_0 x_conv2.O_Core_3
 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
 y_Transfer(conv1.W_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_6821434528920366015
 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_4083612102222981683
 y_Transfer(conv1.I_->_['conv1.part0',_'conv1.part1'])_-3553201336870363689
 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_7934457221615677481
 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_4083612102222981683
 y_Transfer(conv2.W_->_['conv2.part0',_'conv2.part1'])_-7086353137520695497
 y_Transfer(conv2.part0.O_->_['conv2.O'])_4083612102222981683
 y_Transfer(conv2.part0.O_->_['conv2.O'])_5740354900026072187
 y_Transfer(conv2.part0.O_->_['conv2.O'])_-7086353137520695497
 y_Transfer(conv2.part1.O_->_['conv2.O'])_7934457221615677481
 y_Transfer(conv2.part1.O_->_['conv2.O'])_4083612102222981683
 y_Transfer(conv2.part1.O_->_['conv2.O'])_5740354900026072187
 y_Transfer(conv1.part0.O_->_['conv2.part0',_'conv2.part1'])_-5089572303829371927
 y_Transfer(conv1.part1.O_->_['conv2.part0',_'conv2.part1'])_-223774253941714869
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_0
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_0
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_1
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_1
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_2
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_2
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_3
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_3
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_4
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_4
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_5
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_5
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_6
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_6
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_7
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_7
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_8
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_8
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_9
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_9
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_10
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_10
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_11
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_11
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_12
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_12
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_13
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_13
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_14
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_14
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_15
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_15
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_16
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_16
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_17
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_17
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_18
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_18
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_19
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_19
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_20
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_20
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_21
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_21
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_22
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_22
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_23
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_23
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_24
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_24
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_25
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_25
 idleS_CommunicationLink(Core(3),_Core(0),_bw=32)_26
 idleE_CommunicationLink(Core(3),_Core(0),_bw=32)_26
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_0
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_0
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_1
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_1
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_2
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_2
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_3
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_3
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_4
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_4
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_5
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_5
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_6
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_6
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_7
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_7
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_8
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_8
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_9
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_9
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_10
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_10
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_11
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_11
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_12
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_12
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_13
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_13
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_14
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_14
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_15
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_15
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_16
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_16
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_17
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_17
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_18
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_18
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_19
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_19
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_20
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_20
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_21
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_21
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_22
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_22
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_23
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_23
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_24
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_24
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_25
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_25
 idleS_CommunicationLink(Core(1),_Core(2),_bw=32)_26
 idleE_CommunicationLink(Core(1),_Core(2),_bw=32)_26
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_0
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_0
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_1
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_1
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_2
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_2
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_3
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_3
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_4
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_4
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_5
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_5
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_6
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_6
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_7
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_7
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_8
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_8
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_9
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_9
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_10
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_10
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_11
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_11
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_12
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_12
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_13
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_13
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_14
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_14
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_15
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_15
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_16
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_16
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_17
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_17
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_18
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_18
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_19
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_19
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_20
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_20
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_21
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_21
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_22
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_22
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_23
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_23
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_24
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_24
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_25
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_25
 idleS_CommunicationLink(Core(1),_Core(0),_bw=32)_26
 idleE_CommunicationLink(Core(1),_Core(0),_bw=32)_26
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_0
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_0
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_1
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_1
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_2
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_2
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_3
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_3
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_4
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_4
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_5
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_5
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_6
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_6
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_7
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_7
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_8
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_8
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_9
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_9
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_10
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_10
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_11
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_11
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_12
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_12
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_13
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_13
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_14
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_14
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_15
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_15
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_16
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_16
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_17
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_17
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_18
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_18
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_19
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_19
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_20
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_20
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_21
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_21
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_22
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_22
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_23
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_23
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_24
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_24
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_25
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_25
 idleS_CommunicationLink(Core(2),_Core(1),_bw=32)_26
 idleE_CommunicationLink(Core(2),_Core(1),_bw=32)_26
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_0
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_0
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_1
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_1
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_2
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_2
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_3
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_3
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_4
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_4
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_5
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_5
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_6
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_6
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_7
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_7
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_8
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_8
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_9
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_9
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_10
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_10
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_11
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_11
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_12
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_12
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_13
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_13
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_14
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_14
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_15
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_15
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_16
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_16
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_17
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_17
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_18
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_18
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_19
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_19
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_20
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_20
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_21
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_21
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_22
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_22
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_23
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_23
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_24
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_24
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_25
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_25
 idleS_CommunicationLink(Core(6),_Any,_bw=64)_26
 idleE_CommunicationLink(Core(6),_Any,_bw=64)_26
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_0
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_0
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_1
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_1
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_2
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_2
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_3
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_3
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_4
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_4
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_5
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_5
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_6
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_6
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_7
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_7
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_8
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_8
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_9
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_9
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_10
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_10
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_11
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_11
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_12
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_12
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_13
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_13
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_14
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_14
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_15
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_15
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_16
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_16
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_17
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_17
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_18
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_18
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_19
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_19
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_20
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_20
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_21
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_21
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_22
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_22
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_23
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_23
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_24
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_24
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_25
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_25
 idleS_CommunicationLink(Core(0),_Core(3),_bw=32)_26
 idleE_CommunicationLink(Core(0),_Core(3),_bw=32)_26
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_0
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_0
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_1
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_1
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_2
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_2
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_3
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_3
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_4
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_4
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_5
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_5
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_6
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_6
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_7
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_7
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_8
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_8
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_9
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_9
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_10
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_10
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_11
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_11
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_12
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_12
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_13
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_13
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_14
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_14
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_15
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_15
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_16
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_16
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_17
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_17
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_18
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_18
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_19
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_19
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_20
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_20
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_21
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_21
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_22
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_22
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_23
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_23
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_24
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_24
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_25
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_25
 idleS_CommunicationLink(Core(2),_Core(3),_bw=32)_26
 idleE_CommunicationLink(Core(2),_Core(3),_bw=32)_26
Generals
 L_0 L_1 L_2 L_3 L_4 L_5 L_6 L_7 L_8 L_9 L_10 L_11 L_12 L_13 L_14 L_15 L_16
 L_17 L_18 L_19 L_20 L_21 L_22 L_23 L_24 L_25 L_26 overlap total_latency
End
