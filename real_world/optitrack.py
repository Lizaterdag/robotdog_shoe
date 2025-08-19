
# coding: utf-8
"""Command-line NatNet client application for testing.

Copyright (c) 2017, Matthew Edwards.  This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/mje-nz/python_natnet/blob/master/LICENSE.
No part of python_natnet, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""

# NatNet imports
from __future__ import print_function

import argparse
import time
import os
import csv

import attr

import natnet
from scipy.spatial.transform import Rotation

file_extension = str(time.ctime().replace(' ', '-'))

# Other imports
import array as arr


@attr.s
class ClientApp(object):
    
    _client = attr.ib()
    _quiet = attr.ib()

    _last_printed = attr.ib(0)

    # optiData = attr.ib([])
    velocData = []
    
    @classmethod
    def connect(cls, server_name, rate, quiet):
        # print('IN CONNECT')
        print(cls)
        print(server_name)
        print(rate)
        print(quiet)
        if server_name == 'fake':
            client = natnet.fakes.SingleFrameFakeClient.fake_connect(rate=rate)
        else:
            client = natnet.Client.connect("192.168.1.132")
            print('client connected')
        if client is None:
            return None
        return cls(client, quiet)

    def run(self):

        if self._quiet:
            self._client.set_callback(self.callback_quiet)
        else:
            self._client.set_callback(self.callback)
        
        
 
    def run_callback(self):
        self._client.run_once() # changed from .spin() to stop from constantly running

    def callback(self, rigid_bodies, markers, timing):
        """
        :type rigid_bodies: list[RigidBody]
        :type markers: list[LabelledMarker]
        :type timing: TimestampAndLatency
        """
        # print('IN CALLBACK')
        #print()
        #print('{:.1f}s: Received mocap frame'.format(timing.timestamp))
        self.optiData = [] # ADDED THIS, use array in case mutliple rigid bodies

        if rigid_bodies:
            #print('Rigid bodies:')
            for b in rigid_bodies:
                
                # print('\t Id {}: ({: 5.2f}, {: 5.2f}, {: 5.2f}), ({: 5.2f}, {: 5.2f}, {: 5.2f}, {: 5.2f})'.format(
                #     b.id_, *(b.position + b.orientation)
                # ))
                
                # file_path = './' + 'mocap-data' + '/' + file_extension

                # with open(os.path.join(file_path,
                #         'data.csv'), 'a') as fd:
                #     cwriter = csv.writer(fd)
                #     cwriter.writerow([time.time(), b.id_, *(b.position + b.orientation)]) # time.time() is time since 'epoch' - Jan 1 1970 00:00:00
                self.optiData.append([time.time(), b.id_, *(b.position + b.orientation)])
                # print('Current Data ', self.optiData)
                       
        
        '''''
        if markers:
            print('Markers')
            for m in markers:
                print('\t Model {} marker {}: size {:.4f}mm, pos ({: 5.2f}, {: 5.2f}, {: 5.2f}), '.format(
                    m.model_id, m.marker_id, 1000*m.size, *m.position
                ))
        print('\t Latency: {:.1f}ms (system {:.1f}ms, transit {:.1f}ms, processing {:.2f}ms)'.format(
            1000*timing.latency, 1000*timing.system_latency, 1000*timing.transit_latency,
            1000*timing.processing_latency
        ))
        '''''

    def callback_quiet(self, *_):
        #print('IN CALLBACK QUIET')
        if time.time() - self._last_printed > 1:
            print('.')
            self._last_printed = time.time()
        return
    

class Optitrack:

    def __init__(self):
        self.optiCoordRange = 5
        self.optiCoordMin = 0
        self.optiAngRange = 360
        self.optiAngMin = -180 # converted to euler so -180 to 180
    
        parser = argparse.ArgumentParser()
        parser.add_argument('--server', help='Will autodiscover if not supplied')
        parser.add_argument('--fake', action='store_true',
                            help='Produce fake data at `rate` instead of connecting to actual server')
        parser.add_argument('--rate', type=float, default=10,
                            help='Rate at which to produce fake data (Hz)')
        parser.add_argument('--quiet', action='store_true')
        self.args = parser.parse_args()


        folder = 'mocap-data'
        file_path = './' + folder + '/' + file_extension

        # Create experiment folder
        # if not os.path.exists(file_path):
        #    os.makedirs(file_path)

        
        # with open(os.path.join(file_path,
        #         'data.csv'), 'a') as fd:
        #     cwriter = csv.writer(fd)
        #     cwriter.writerow(['Timestamp', 'ID', 'Pos x', 'Pos y', 'Pos z', 'Quat w', 'Quat x', 'Quat y', 'Quat z']) # time.time() is time since 'epoch' - Jan 1 1970 00:00


        self.app = ClientApp.connect('fake' if self.args.fake else self.args.server, self.args.rate, self.args.quiet)
        self.app.run() # set up callback

    def optiTrackGetPos(self):
        try:
            self.app.run_callback()
            # extract elements of data 
            coord = self.app.optiData[0][2:5] # x y z
            #normalizedOptiCoord = [2*(pos-self.optiCoordMin)/self.optiCoordRange-1 for pos in coord]
            normalizedOptiCoord = coord
            # print('COORD',coord)
            
            quaternion = self.app.optiData[0][5:] # quat w, quat x, quat y, quat z
            # print('QUAT', quaternion)
            quaternion.append(quaternion.pop(0)) # put into quat x, quat y, quat z, quat w form
            
            # convert quaternion
            rotation = Rotation.from_quat(quaternion) # x y z w
            # eulerRot = rotation.as_euler('xyz', degrees=True) # returning in degrees
            # normalizedOptiAngle = [2*(pos-self.optiAngMin)/self.optiAngRange-1 for pos in eulerRot]
            normalizedOptiAngle = rotation.as_euler('xzy', degrees=True)
            
            #result = [*coord, *eulerRot]
            result = [*normalizedOptiCoord, *normalizedOptiAngle] # return normalized data
         
            return normalizedOptiCoord, normalizedOptiAngle #eulerRot
            
        except natnet.DiscoveryError as e:
            print('Error:', e)



if __name__ == '__main__': # test
    opt = Optitrack()
    opt.optiTrackGetPos()