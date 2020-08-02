import React, {Component} from 'react';
import {
    NativeModules,
    PanResponder,
    Dimensions, 
    Image, 
    View, 
    Animated} from 'react-native';

import Svg, {Polygon} from 'react-native-svg';

const AnimatedPolygon = Animated.createAnimatedComponent(Polygon);
const dimensions = Dimensions.get('window');

class Cropper extends Component{
    constructor(props){
        super(props);
        //the X and Y range assumt centering of the cropper
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

    createPanResponser(corner) {
        return PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onPanResponderMove: (e, gesture) => { 
                console.log(gesture);
                console.log(this.props.minX);
                console.log(this.props.minY);
                console.log(this.props.maxX);
                console.log(this.props.maxY);
                if(gesture.moveX >= this.props.minX && gesture.moveX <= this.props.maxX && gesture.moveY >= this.props.minY && gesture.moveY <= this.props.maxY){
                    console.log('coords');
                    console.log(gesture.moveX);
                    console.log(gesture.moveY);
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

    

    render(){
        return(
            <View>
                <View
                    style = {{
                        height :  this.props.viewHeight+this.props.viewPadding,
                        width : this.props.viewWidth+this.props.viewPadding,
                    }}>
                    <Image
                        source = {{uri : this.props.initialImage}}
                        style = {{
                            height : this.props.viewHeight,
                            width : this.props.viewWidth
                        }}/>
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
                            s(this.props).handler,
                        ]}
                    >
                        <View
                            style={[
                                s(this.props).handlerI,
                                { left: -10, top: -10 },
                            ]}
                        />
                        <View
                            style={[
                                s(this.props).handlerRound,
                                { left: 31, top: 31 },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderTopRight.panHandlers}
                        style={[
                            this.state.topRight.getLayout(),
                            s(this.props).handler,
                        ]}
                    >
                        <View
                            style={[
                                s(this.props).handlerI,
                                { left: 10, top: -10 },
                            ]}
                        />
                        <View
                            style={[
                                s(this.props).handlerRound,
                                { right: 31, top: 31 },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderBottomLeft.panHandlers}
                        style={[
                            this.state.bottomLeft.getLayout(),
                            s(this.props).handler,
                        ]}
                    >
                        <View
                            style={[
                                s(this.props).handlerI,
                                { left: -10, top: 10 },
                            ]}
                        />
                        <View
                            style={[
                                s(this.props).handlerRound,
                                { left: 31, bottom: 31 },
                            ]}
                        />
                    </Animated.View>
                    <Animated.View
                        {...this.panResponderBottomRight.panHandlers}
                        style={[
                            this.state.bottomRight.getLayout(),
                            s(this.props).handler,
                        ]}
                    >
                        <View
                            style={[
                                s(this.props).handlerI,
                                { left: 10, top: 10 },
                            ]}
                        />
                        <View
                            style={[
                                s(this.props).handlerRound,
                                { right: 31, bottom: 31 },
                            ]}
                        />
                    </Animated.View>
                </View>
            </View>
        );
    }
}

const s = (props) => ({
    handlerI: {
        borderRadius: 0,
        height: 20,
        width: 20,
        backgroundColor: props.handlerColor || 'blue',
    },
    handlerRound: {
        width: 39,
        position: 'absolute',
        height: 39,
        borderRadius: 100,
        backgroundColor: props.handlerColor || 'blue',
    },
    image: {
        width: props.viewWidth,
        position: 'absolute',
    },
    bottomButton: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'blue',
        width: 70,
        height: 70,
        borderRadius: 100,
    },
    handler: {
        height: 140,
        width: 140,
        overflow: 'visible',
        marginLeft: -70,
        marginTop: -70,
        alignItems: 'center',
        justifyContent: 'center',
        position: 'absolute',
    },
    cropContainer: {
        position: 'absolute',
        left: 0,
        width: props.viewWidth,
        top: 0,
    },
});

export default Cropper;
