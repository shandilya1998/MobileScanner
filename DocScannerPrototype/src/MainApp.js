import React, {Component} from 'react';
import {createDrawerNavigator } from '@react-navigation/drawer';
import {NavigationContainer} from '@react-navigation/native';
import {navigatorRef} from './rootNavigator';
import MobileScanner from './containers/MobileScanner';

MainStack = createDrawerNavigator();

class MainApp extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <NavigationContainer ref = {navigatorRef}>
                <MainStack.Navigator initialRouteName = {'Scanner'}>
                    <MainStack.Screen 
                        name = {'Scanner'}
                        component = {MobileScanner}/>
                </MainStack.Navigator>
            </NavigationContainer>
        );
    }
}

export default MainApp;
