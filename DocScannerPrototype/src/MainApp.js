import React, {Component} from 'react';
import {Platform} from 'react-native';
import {createDrawerNavigator } from '@react-navigation/drawer';
import {NavigationContainer} from '@react-navigation/native';
import {navigatorRef} from './rootNavigator';
import MobileScanner from './containers/MobileScanner';
import ScanContainer from './containers/ScanContainer';
import {checkMultiple, PERMISSIONS} from 'react-native-permissions';

MainStack = createDrawerNavigator();

class MainApp extends Component{
    constructor(props){
        super(props);
    }

    getPermissions(){
        if(Platform.OS == 'android'){
            
        }
    }

    render(){
        return(
            <NavigationContainer ref = {navigatorRef}>
                <MainStack.Navigator 
                    initialRouteName = {'Scanner'}
                    options = {{swipeEnabled : false,}}>
                    <MainStack.Screen 
                        name = {'Scanner'}
                        component = {ScanContainer}/>
                </MainStack.Navigator>
            </NavigationContainer>
        );
    }
}

export default MainApp;
