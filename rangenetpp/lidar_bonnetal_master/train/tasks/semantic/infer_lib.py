#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import shlex
import pdb

import sys

sys.path.insert(0, '../')


from .modules.user import *

def main(arg_string = None):
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + '/',
      help='Directory to put the predictions. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--model', '-m',
      type=str,
      required=True,
      default=None,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--dump', '-u',
      type=str,
      required=False,
      default=None,
      help='LiDAR Data Dump Folder'
  )
  parser.add_argument(
      '--kitti', '-k',
      required=False,
      default=False,
      help='Whether to dump KITTI features',
      action="store_true"
  )
  parser.add_argument(
      '--frd_dir',
      type=str,
      required=False,
      default=None,
      help='Place to dump FRD features'
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      required=False,
      default=None,
      help='Place to dump predictions'
  )

  parser.add_argument(
      '--kitti_count',
      type=int,
      required=False,
      default=20,
      help='Number of KITTI samples to generate'
  )

  parser.add_argument(
      '--seed',
      type=int,
      required=False,
      default=0,
      help='Seed for KITTI'
  )

  parser.add_argument(
      '--point_cloud',
      required=False,
      default=False,
      action='store_true',
      help='Use a folder of point clouds'
  )

  FLAGS, unparsed = None, None
  if(arg_string is None):
    FLAGS, unparsed = parser.parse_known_args()
  else:
    FLAGS = parser.parse_args(shlex.split(arg_string))

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("log", FLAGS.log)
  print("model", FLAGS.model)
  print("----------\n")
  #print("Commit hash (training version): ", str(
  #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file from %s" % FLAGS.model)
    
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % FLAGS.model)
    DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
    os.makedirs(os.path.join(FLAGS.log, "sequences"))
    for seq in DATA["split"]["train"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["valid"]:
      seq = '{0:02d}'.format(int(seq))
      print("valid", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      print("test", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if os.path.isdir(FLAGS.model):
    print("model folder exists! Using model from %s" % (FLAGS.model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()

  # create user and infer dataset
  user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.dump, FLAGS.kitti, FLAGS.frd_dir, FLAGS.output_dir, FLAGS.kitti_count, FLAGS.seed, FLAGS.point_cloud)
  user.infer()


  
