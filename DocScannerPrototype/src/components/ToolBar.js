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
                        inactiveSlideScale = {0.75}
                        sliderWidth = {this.props.width-2*this.props.margin}
                        firstItem = {(this.props.tools.length-1)/2}
                        onSnapToItem = {this.props.onSnapToItem}/>
            </View>
        );
    }
}

export default ToolBar;
