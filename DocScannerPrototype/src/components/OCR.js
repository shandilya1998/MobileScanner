import React, {Component} from 'react';
import {
    NativeModules,
    PanResponder,
    Dimensions, 
    Image, 
    View, 
    Animated} from 'react-native';
import {styles} from '../assets/styles';
import Svg, {Polygon} from 'react-native-svg';

const dimensions = Dimensions.get('window');
