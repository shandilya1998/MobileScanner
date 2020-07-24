import React, {Component} from 'react';
import {View, Text} from 'react-native';
import Carousel from 'react-native-snap-carousel';

class Library extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <View
                style = {styles.container}>
                <View
                    style = {{
                        flex : 1,
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                    <Carousel/>
                </View>
                <View 
                    style = {{
                        flex : 1, 
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                </View>
            </View>
        );
    }
}

export default Library;
