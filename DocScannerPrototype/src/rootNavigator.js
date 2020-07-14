import React from 'react';
import { DrawerActions } from '@react-navigation/native';

export const navigatorRef = React.createRef();

export function navigate(name, params = {}){
        //console.log(params);
        //console.log('empulseflashbacknavigator')
        navigatorRef.current?.navigate(name, params);
}

export function goBack(){
        navigatorRef.current?.goBack();
}
export function openDrawer(){
    navigatorRef.current?.dispatch(DrawerActions.openDrawer());
} 
