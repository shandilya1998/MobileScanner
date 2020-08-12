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

const AnimatedPolygon = Animated.createAnimatedComponent(Polygon);
const dimensions = Dimensions.get('window');

class Cropper extends Component{
    constructor(props){
        super(props);
        //the X and Y range assumt centering of the cropper
        this.imageCoordinatesToViewCoordinates = this.imageCoordinatesToViewCoordinates.bind(this);
        this.viewCoordinatesToImageCoordinates = this.viewCoordinatesToImageCoordinates.bind(this);
        this.createPanResponser = this.createPanResponser.bind(this);
        this.state = {
            moving : false,
            topLeft: new Animated.ValueXY(
                props.rectangleCoordinates
                    ? this.imageCoordinatesToViewCoordinates(
                          props.rectangleCoordinates.topLeft,
                          true,
                      )   
                    : { x: 100, y: 100 },
            ),  
            topRight: new Animated.ValueXY(
                props.rectangleCoordinates
                    ? this.imageCoordinatesToViewCoordinates(
                          props.rectangleCoordinates.topRight,
                          true,
                      )   
                    : { x: this.props.viewWidth - 100, y: 100 },
            ),  
            bottomLeft: new Animated.ValueXY(
                props.rectangleCoordinates
                    ? this.imageCoordinatesToViewCoordinates(
                          props.rectangleCoordinates.bottomLeft,
                          true,
                      )   
                    : { x: 100, y: this.props.viewHeight - 100 },
            ),  
            bottomRight: new Animated.ValueXY(
                props.rectangleCoordinates
                    ? this.imageCoordinatesToViewCoordinates(
                          props.rectangleCoordinates.bottomRight,
                          true,
                      )   
                    : { 
                          x: this.props.viewWidth - 100,
                          y: this.props.viewHeight - 100,
                      },  
            ),
        } 
        this.state = {
            ...this.state, 
            overlayPositions: `${this.state.topLeft.x._value},${
                this.state.topLeft.y._value
            } ${this.state.topRight.x._value},${this.state.topRight.y._value} ${
                this.state.bottomRight.x._value
            },${this.state.bottomRight.y._value} ${
                this.state.bottomLeft.x._value
            },${this.state.bottomLeft.y._value}`,
        };
        this.panResponderTopLeft = this.createPanResponser(this.state.topLeft);
        this.panResponderTopRight = this.createPanResponser(
            this.state.topRight,
        );  
        this.panResponderBottomLeft = this.createPanResponser(
            this.state.bottomLeft,
        );  
        this.panResponderBottomRight = this.createPanResponser(
            this.state.bottomRight,
        ); 
    }
    /* 
    componentDidUpdate(){
        console.log('top left');
        console.log(this.state.topLeft);
        console.log('top right');
        console.log(this.state.topRight);
        console.log('bottom left');
        console.log(this.state.bottomLeft);
        console.log('bottom right');
        console.log(this.state.bottomRight);
    }
    */
    createPanResponser(corner) {
        return PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onPanResponderMove: (e, gesture) => { 
                //console.log('minX');
                //console.log(this.props.minX);
                //console.log('minY');
                //console.log(this.props.minY);
                //console.log('maxX');
                //console.log(this.props.maxX);
                //console.log('maxY');
                //console.log(this.props.maxY);
                if(gesture.moveX >= this.props.minX && gesture.moveX <= this.props.maxX && gesture.moveY >= this.props.minY && gesture.moveY <= this.props.maxY){
                    return(
                        Animated.event([
                            null,
                            {
                                dx: corner.x,
                                dy: corner.y,
                            },
                        ])(e, gesture)
                    );
                }
            },
            onPanResponderRelease: () => {
                corner.flattenOffset();
                this.updateOverlayString();
            },
            onPanResponderGrant: () => {
                corner.setOffset({ x: corner.x._value, y: corner.y._value });
                corner.setValue({ x: 0, y: 0 });
            },
        });
    }

    crop() {
        const coordinates = { 
            topLeft: this.viewCoordinatesToImageCoordinates(
                this.state.topLeft
            ),            
            topRight: this.viewCoordinatesToImageCoordinates(
                this.state.topRight,
            ),      
            bottomLeft: this.viewCoordinatesToImageCoordinates(
                this.state.bottomLeft,
            ),      
            bottomRight: this.viewCoordinatesToImageCoordinates(
                this.state.bottomRight,
            ),      
            height: this.props.height,
            width: this.props.width,
        };     
        console.log('view to image coords');
        console.log(coordinates);
        NativeModules.CustomCropManager.crop(
            coordinates,
            this.props.initialImage,
            (err, res) => this.props.updateImage(res.image, coordinates),
        );      
    }

    updateOverlayString() {
        this.setState({
            overlayPositions: `${this.state.topLeft.x._value},${
                this.state.topLeft.y._value
            } ${this.state.topRight.x._value},${this.state.topRight.y._value} ${
                this.state.bottomRight.x._value
            },${this.state.bottomRight.y._value} ${
                this.state.bottomLeft.x._value
            },${this.state.bottomLeft.y._value}`,
        });
    }

    imageCoordinatesToViewCoordinates(corner){
        return {
            x : (corner.x*this.props.viewWidth)/this.props.width,
            y : (corner.y*this.props.viewHeight)/this.props.height, 
        };
    }

    viewCoordinatesToImageCoordinates(corner){
        return {
            x : ((corner.x._value)/this.props.viewWidth)*this.props.width,
            y : ((corner.y._value)/this.props.viewHeight)*this.props.height,
        };
    }

    render(){
        return(
            <View
                style = {{
                    flex : 1,
                    alignItems : 'center',
                    justifyContent : 'flex-end',
                }}>
                <View
                    style = {[
                        styles.cropContainer,
                        {
                            height :  this.props.viewHeight,
                            width : this.props.viewWidth,
                        }]}>
                    <Image
                        source = {{uri : this.props.initialImage}}
                        style = {[
                            styles.images,
                            {
                                height : this.props.viewHeight,
                                width : this.props.viewWidth
                            }]}/>
                    <Svg
                        height={this.props.viewHeight}
                        width={this.props.viewWidth}
                        style={{ position: 'absolute', left: 0, top: 0 }}
                    >
                        <AnimatedPolygon
                            ref={(ref) => (this.polygon = ref)}
                            fill={this.props.overlayColor || 'blue'}
                            fillOpacity={this.props.overlayOpacity || 0.5}
                            stroke={this.props.overlayStrokeColor || 'blue'}
                            points={this.state.overlayPositions}
                            strokeWidth={this.props.overlayStrokeWidth || 3}
                        />
                    </Svg>
                    <Animated.View
                        {...this.panResponderTopLeft.panHandlers}
                        style={[
                            this.state.topLeft.getLayout(),
                            styles.handler,
                        ]}
                    >
                        <View
                            style={[
                                styles.handlerI,
                                { 
                                    left: -10, 
                                    top: -10,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                        <View
                            style={[
                                styles.handlerRound,
                                { 
                                    left: 31, 
                                    top: 31,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderTopRight.panHandlers}
                        style={[
                            this.state.topRight.getLayout(),
                            styles.handler,
                        ]}
                    >
                        <View
                            style={[
                                styles.handlerI,
                                { 
                                    left: 10, 
                                    top: -10,
                                    backgroundColor: this.props.handlerColor || 'blue' },
                            ]}
                        />
                        <View
                            style={[
                                styles.handlerRound,
                                { 
                                    right: 31, 
                                    top: 31,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderBottomLeft.panHandlers}
                        style={[
                            this.state.bottomLeft.getLayout(),
                            styles.handler,
                        ]}
                    >
                        <View
                            style={[
                                styles.handlerI,
                                { 
                                    left: -10, 
                                    top: 10,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                        <View
                            style={[
                                styles.handlerRound,
                                { 
                                    left: 31, 
                                    bottom: 31,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderBottomRight.panHandlers}
                        style={[
                            this.state.bottomRight.getLayout(),
                            styles.handler,
                        ]}
                    >
                        <View
                            style={[
                                styles.handlerI,
                                { 
                                    left: 10, 
                                    top: 10,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                        <View
                            style={[
                                styles.handlerRound,
                                { 
                                    right: 31, 
                                    bottom: 31,
                                    backgroundColor: this.props.handlerColor || 'blue',
                                },
                            ]}
                        />
                    </Animated.View>
                </View>
            </View>
        );
    }
}

export default Cropper;
