import React, {Component} from 'react';
import {createStackNavigator} from '@react-navigation/stack';
import MobileScanner from '../MobileScanner';
import EditScreen from '../EditScreen';
import SaveScreen from '../SaveScreen';
import {NavigationContainer} from '@react-navigation/native';
import {styles} from '../../assets/styles';
const Stack = createStackNavigator();

class ScanContainer extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <NavigationContainer independent = {true}>
                <Stack.Navigator 
                    initialRouteName = {'scan'}
                    screenOptions = {{headerShown : false}}>
                    <Stack.Screen 
                        name = 'scan' 
                        component = {MobileScanner}/>
                    <Stack.Screen
                        name = 'edit'
                        component = {EditScreen}/>
                    <Stack.Screen
                        name = 'saved'
                        component = {SaveScreen}/>
                </Stack.Navigator>
            </NavigationContainer>
        );
    }
}

export default ScanContainer;
