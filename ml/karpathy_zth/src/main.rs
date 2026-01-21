mod value;
use crate::value::Value;

fn main() {
    let a = Value::new_data(2.0);
    let b = Value::new_data(-3.0);
    let c = Value::new_data(10.0);
    let d = a*b + c;
    println!("{:?}", d);
}
