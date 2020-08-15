import React, {Component} from 'react';
import {View} from 'react-native';
import Carousel, { Pagination } from 'react-native-snap-carousel';

class ToolBar extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <View 
                style = {[
                    {
                        flex : 1,
                        //width : this.props.width,
                    },
                    this.props.containerStyle
                ]}>
                <Carousel
                    renderItem = {({item, index}) => this.props.renderItem({item, index})}
                    data = {this.props.tools}
                    vertical = {false}
                    itemWidth = {85}
                    sliderWidth = {this.props.width}
                    firstItem = {(this.props.tools.length-1)/2}/>
            </View>
        );
    }
}

export default ToolBar;
