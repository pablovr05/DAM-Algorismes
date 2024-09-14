package com.project;

public abstract class Observable<T> {

    private T value;

    public Observable(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        T oldValue = this.value;
        this.value = value;
        this.propertyChange(oldValue, value);
    }

    public abstract void propertyChange(T oldValue, T newValue);
}