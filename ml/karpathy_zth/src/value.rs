use std::ops::{Add, Mul};

#[derive(Debug, Default)]
pub struct Value {
    data: f64,

    // in order to form the DAG, each time an operation is done, we need to track:
    // - the two input values
    // - the operation done to them

    // this then allows us to derive, at each step, the expression "linking" together the output and the input
    children: Vec<Value>,
    op: String
}

impl Value {
    pub fn new_data(data: f64) -> Self {
        return Self {
            data: data,
            ..Default::default()
        }
    }

    pub fn new_op(data: f64, op: String) -> Self {
        return Self {
            data: data,
            op: op,
            ..Default::default()
        }
    }

    pub fn new(data: f64, op: String, children: Vec<Value>) -> Self {
        return Self {
            data: data,
            op: op,
            children: children
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self {
        return Value::new(self.data + other.data, "+".to_string(), vec![self, other])
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self {
        return Value::new(self.data * other.data, "*".to_string(), vec![self, other])
    }
}